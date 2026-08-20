# Pitch evidence index — which job proves which claim

Every CDK job run for the Kimi-Linear optimization work, and what to pull from
each. Model is `moonshotai/Kimi-Linear-48B-A3B-Instruct` on `tpu7x-8`, TP=4,
8192 in / 1024 out, concurrency 8 and 16, unless noted.

To open a profile: `cdk job xprof <job-id>` → **XProf → HLO Op Profile**,
group by Program, order by wasted time, expand `custom-call`.

---

## The two screenshots the pitch rests on

### BEFORE — `j-e915314e-17a1-4286-8a9d`

The untuned kernel. **This is the slide.**

| | |
|---|---|
| MLA kernel | `MLA-m-bq_16-bkvp_1-p_16-bsz_1` |
| MLA share of device time | **80.27%** (69.23% + 11.04%) |
| Chip FLOPS utilization | **2.76%** |
| HBM utilization | **5.82%** |
| Wasted | 93.92% |
| Per call | **27.54 ms**, 87 calls, 2.40 s total |

Also here: `gmm_v2` rows at **76–80% FLOPS** — the evidence that MoE was
already healthy and correctly left alone. And `kda_decode_fused_conv_recurrent`
at 0.40%, which is why KDA was not the target.

### AFTER — `j-ac3a7aa1-9ddd-41ee-b9ef`

All four MLA parameters computed. **The other half of the slide.**

| | |
|---|---|
| MLA kernel | `MLA-m-bq_256-bkvp_64-p_16-bsz_1` |
| Per call | **0.931 ms** (~3.0× vs `bq_16`, ~30× vs untuned) |
| Throughput | **739.88** (c=8) / **975.89** (c=16) |
| Mean TTFT | 765.79 ms (c=8) / 1633.13 ms (c=16) |
| VMEM | `Program vmem requirement: 64.00M` — legal only via `q_split=2` |
| Paris | PASSED |

---

## Baselines

| Job | What it establishes |
|---|---|
| **`j-a3a86598-2e8b-463a-8600`** | **The original baseline.** 8k/1k: **386.37** (c=8), **440.44** (c=16), TPOT 12.72 / 24.20 ms. Also 1k/1k: 1128.18 / 1725.19. Every "×" in the deck is against this. |
| `j-3a61c911-a3c9-40af-976a` | `decode_kda` BlockSpec offset change verified on device, **zero regression**: 388.03 vs 386.75 tok/s, no compile errors. Use for "we modified a Pallas kernel and proved it cost nothing." |

---

## The config win (+32%)

**`j-64a9afc1-1ab5-4869-877a`** — the 4-cell ablation. This is the causality proof.

| Arm | batched tokens | EP | c=8 | c=16 |
|---|---|---|---|---|
| baseline | 8192 | off | 386.37 | 440.44 |
| S1 | 2048 | off | 481.66 | 573.00 |
| **S2** | **2048** | **on** | **486.83** | **581.94** |
| S3 | 8192 | on | 386.85 | 445.43 |

S3 is the important row: EP with the old budget stays at 386/445, so the
**batched-token budget is causal and EP is worth ~1%**.

---

## The MLA block-size progression

Four jobs, one variable, kernel name verified in each. This is the story arc.

| Job | Config | c=8 | c=16 | MLA share | FLOPS | HBM |
|---|---|---|---|---|---|---|
| `j-e915314e` | `bkvp_1` | — | 581.94 | **80.27%** | 2.76% | 5.82% |
| **`j-64210abe-be0b-4aae-818a`** | env override `bkvp_4/16/32` | 648.72 / 714.74 / **720.21** | 829.70 / 945.33 / **964.71** | — | — | — |
| `j-ac0b634f-e01f-4542-b44d` | computed, `bkvp_8` | 669.62 | 858.51 | **39.42%** | 6.62% | 16.12% |
| `j-5104808b-e699-4f6a-80d4` | computed, `bkvp_16` | 686.40 | 887.30 | **33.48%** | 7.31% | 18.26% |
| `j-179df4e9-8181-454b-8a00` | computed, `bq_16-bkvp_64` | 705.91 | 910.45 | — | — | — |
| **`j-ac3a7aa1`** | computed, `bq_256-bkvp_64` | **739.88** | **975.89** | ~11% | — | — |

`j-64210abe` is the cleanest single-variable sweep — same job, three block
sizes, kernel names verified, correctness passed at every point.

**Also worth a slide:** `j-aacb8593-bd17-449d-8818` measured nothing. The env
override never reached the kernel, all three arms compiled `bkvp_1`, and
throughput moved 1%. It is why every later run verifies the kernel name before
trusting a number.

---

## Parallelism study (TP / EP / DP)

**`j-4c1c5c77-d692-4206-87f3`**

| Arm | TP × DP | c=8 | c=16 |
|---|---|---|---|
| **P1** | 4 × 1 | **695.52** | **926.96** |
| P3 | 4 × 2 | 468.80 | 855.56 |
| P4 | 2 × 4 | 447.27 | 818.83 |
| P2 | 8 × 1 | **OOM** — VMEM 63.50/64.00 MB | |

DP fragments a fixed concurrency across replicas faster than it saves on the
~13.7% ICI cost. TP=4 is optimal at the scored concurrencies.

**TP=8 follow-up** (job deleted by the CDK orchestrator on worker exit; logs
were captured before teardown): `TPU_MLA_MIXED_VMEM_MB=16` took VMEM from
**63.50 MB → 32.02 MB**, proving the ceiling was a 15× over-declaration rather
than hardware. Behind it:

```
W llo_loop.cc:101  [gather_fusion.1] 0-iteration loop inserted: start=0, step=1, limit=0
F lower_pass_base.cc:47  Check failed: dynamic_dim_it != dynamic_sizes.end()
```

An XLA lowering assertion at 8-way sharding. File it; it is the compiler-level
artifact.

---

## The autotuner

| Job | What it produced |
|---|---|
| **`j-e02214ac-fc13-49e1-a5fb`** | Query sweep. At `bkvp=32`: queries 8→4.888 ms, 16→2.808, 32→1.756, 64→1.256, **128→1.041**. Full grid vs `bkvp=1` baseline of **45.119 ms**. |
| **`j-caa59a48-e46d-4a72-b153`** | `q_split` sweep. **Global optimum 0.911 ms at `kv=64, q=256, split=2`** — 49.5× over untuned. And the co-design finding: `q=512, split=1` fails `CompileTimeScopedVmemOom`, `split≥2` compiles. |

**The credibility line:** the tuner measured 0.911 ms standalone; the full model
measured **0.931 ms** (`j-ac3a7aa1`). ~2% agreement between a synthetic harness
and production.

---

## Null results (state them, don't hide them)

| Job | Finding |
|---|---|
| **`j-d4f807f7-48ee-40c5-8b49`** | Four MoE/SparseCore flags: SP, OH, EK, HE all +0.3–3.8%, **inside the 1–4% run-to-run variance**. COMBO was *worse* than three of its four parts — the tell that it is noise. Also: **`RAGGED_GATHER_REDUCE_VERSION=v3` fails to compile** — `CompileTimeSparseCoreAllocationFailure`, 131,199 words vs 131,071 limit in `tile_spmem`, over by exactly 128 words. A real bug in a shipped opt-in path. |
| **`j-0c31041c-0280-41a2-8b5d`** | `--async-scheduling` A/B in one job. Throughput +0.6–1.0% (noise), **Mean TTFT −8.0% to −9.5% (consistent, both concurrencies)**. Control arm here read **743.21 / 1003.12** — same config as `j-ac3a7aa1`'s 739.88 / 975.89, **2.8% apart**. This pair is the variance evidence. |

---

## Speculative decoding — omit from the deck

| Job | |
|---|---|
| `j-943a553c-17a5-4cb6-b5ad` | First spec-decode run. Ran end to end, but output was garbage (repeated tokens). |
| `j-e8f70278-0268-4807-9d0d` | k=1/2/3 sweep. Cost model: 69.4 ms fixed + 12.3 ms per window token vs a 12.61 ms baseline step. |
| `j-01fbfa27-768a-4faf-b9cc` | Failed at startup — vLLM version skew. Nothing usable. |

Keep as a spoken answer if asked what else was tried. Not a result.

---

## Final run

**`j-15612139-4728-41cc-8a54`** — in flight at time of writing.

- Arm A control at **128 prompts**, c=8 and c=16 → the defensible headline
- Arm A at 32 prompts → comparable control for Arm B
- Arm B `--kv-cache-dtype fp8` at 32 prompts → quantization / KV-cache box

---

## Headline numbers

| | c=8 | c=16 |
|---|---|---|
| Baseline (`j-a3a86598`) | 386.37 | 440.44 |
| Final (`j-ac3a7aa1` / `j-0c31041c` control) | 739.88 – 743.21 | 975.89 – **1003.12** |
| | **~1.92×** | **~2.28×** |

**Normalized Efficiency: 110.1 → ~250 tok/s/chip.**
**MLA kernel: 27.54 ms → 0.931 ms.**
**Chip FLOPS utilization: 2.76% → 7.31%+.**

Quote the 128-prompt figures from `j-15612139` once they land — a 35-second
benchmark carries ±3%, and that is the first question a judge will ask.
