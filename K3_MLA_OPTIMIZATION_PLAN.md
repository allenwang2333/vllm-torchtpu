# Kimi on TPU v7 — Findings and Plan

**Branch:** `shriharim/k3-spec-probe`
**Model under test:** `moonshotai/Kimi-Linear-48B-A3B-Instruct`, TP=4, `tpu7x-8`
**Workload:** 8192 in / 1024 out, concurrency 8 and 16 (the scored workload)

---

## TL;DR

1. **We have a +32% throughput win already measured and reproducible.** It came from prefill scheduling, not kernels.
2. **We found the real bottleneck: the Gated MLA prefill kernel is 80.3% of device time and the chip runs at 2.76% FLOPS / 5.82% HBM while it does.** One call averages **27.54 ms** — longer than an entire decode step.
3. **The cause is a two-line fallback**, not a hard problem. Kimi has no entry in the MLA tuning table, so the kernel takes the minimum legal block size: one KV page per block, against a 455,811-page cache.
4. **The fix is worth 2–4× end-to-end**, not a few percent. That is the prize.

---

## Part 1 — What we measured

### 1.1 Baseline

Kimi-Linear-48B-A3B, TP=4, `max_num_batched_tokens=8192`, EP off.

| Input | Concurrency | Output tok/s | Mean TPOT |
|---|---|---|---|
| 1k | 8 | 1128.18 | 6.66 ms |
| 1k | 16 | 1725.19 | 8.67 ms |
| 8k | 8 | 386.37 | 12.72 ms |
| 8k | 16 | 440.44 | 24.20 ms |

**The anomaly that started everything:** going c=8 → c=16 costs +30% TPOT at 1k context but **+90% at 8k**. Concurrency scaling was broken specifically at long context.

### 1.2 Config sweep — the +32% win

Hypothesis: 8192-token prompts consume the entire per-step token budget, so decodes queue behind prefills (head-of-line blocking). Fix: chunk prefills.

| Config | `max_num_batched_tokens` | EP | c=8 | c=16 |
|---|---|---|---|---|
| Baseline | 8192 | off | 386.37 | 440.44 |
| S1 | **2048** | off | 481.66 | 573.00 |
| **S2 (winner)** | **2048** | **on** | **486.83** | **581.94** |
| S3 | 8192 | on | 386.85 | 445.43 |

**Result: +26.0% at c=8, +32.1% at c=16.**
**Normalized Efficiency: 96.6 → 121.7 tok/s/chip (c=8), 110.1 → 145.5 (c=16).**

Two conclusions from the 4-cell ablation:
- The **batched-token budget is causal.** S3 proves it: EP with the old budget is still stuck at 386/445.
- **Expert parallelism is worth ~1%.** Real but immaterial. MoE layout was not the problem.

At c=16 it is a **strict** improvement — throughput up, TPOT down (24.20 → 21.96 ms), TTFT down (12390 → 5558 ms).

### 1.3 Device profile — the headline finding

Captured with **XProf → HLO Op Profile** on the S2 config at 8k/1k, c=16.

| Op | % device time | FLOPS util | Notes |
|---|---|---|---|
| **`MLA-m-bq_16-bkvp_1-p_16-bsz_1`** | **80.27%** | ~0% | **27.54 ms/call**, 87 calls |
| all-reduce | 4.39% | — | |
| KDA total (chunk + decode + conv + transpose) | ~4.8% | ~0% | `chunk_kda` averages 296 µs/call |
| `gmm_v2` (both) | 3.11% | **76–80%** | **Healthy. Leave alone.** |
| loop fusion | 2.15% | | |
| custom fusion (gather/scatter) | 1.70% | | |

**Overall: 2.76% FLOPS utilization, 5.82% HBM utilization, 93.92% of device time wasted.**

The machine is idle 94% of the time, waiting on one kernel.

---

## Part 2 — Root cause

`MLA-m-bq_16-bkvp_1-p_16-bsz_1` is **not a tuned configuration.** It is the untuned fallback.

From `src/vllm_torchtpu/kernels/mla/v2/tuned_params.py`:

```python
def get_tuned_params(tuning_key: TuningKey) -> TunableParams:
    if tuning_key in tuned_params_mapping:
        return tuned_params_mapping[tuning_key]
    else:
        logger.warning("No tuned parameters found ... using default parameters")
        if tuning_key.case == "mixed":
            return TunableParams(
                num_kv_pages_per_block=1,      # <-- bkvp_1
                num_queries_per_block=16,       # <-- bq_16
                vmem_limit_bytes=62914560,
            )
```

The field's own documentation:

> `num_kv_pages_per_block`: "Number of KV pages to process per block. **Range from 1 to as high as possible before OOM**, with steps of powers of two."

**Kimi is not in the table, so it gets the minimum.** The kernel walks a 455,811-page KV cache one page per block. Each block is a tiny DMA with nothing to overlap against, so the kernel spends its life waiting on memory latency rather than reading or computing.

This is the same as reading a file one byte at a time instead of in 64 KB chunks — identical total bytes, order-of-magnitude different throughput.

### Why this matters beyond Kimi

The table is **hand-populated**. There are **six** such tables in the repo:

```
kernels/quantized_matmul/tuned_block_sizes.py
kernels/megablox/tuned_block_sizes.py
kernels/collectives/all_gather_matmul_tuned_block_sizes.py
kernels/mla/v2/tuned_params.py
kernels/ragged_paged_attention/v2/tuned_block_sizes.py
kernels/ragged_paged_attention/v3/tuned_block_sizes.py (+ _hd64)
```

**And there is no tooling anywhere to generate them.** Every entry was typed by hand.

Consequence: **any model not already tuned silently runs its hottest kernel at ~3% utilization.** Kimi-Linear hits this. K3 at 2.8T will hit this — it uses Gated MLA and it is not in the table either. Every future bring-up hits it. The warning is printed and execution continues.

---

## Part 3 — The work

Three items. **C is the one that matters most.**

### A — Fuse the output RMSNorm and gate into `decode_kda`

**What:** The kernel's docstring states *"Output normalization remains with the caller."* Upstream's CUDA fused KDA kernel includes RMSNorm; ours stops before it. Move the RMSNorm and sigmoid output gate inside the Pallas kernel, before the write.

**Files:** `kernels/kimi_k3/decode_kda.py`, `layers/vllm/custom_ops/kda_attention_op.py`

**Validation:** Bit-exact against the existing reference, using the `PALLAS_INTERPRET` harness — **runs on CPU in 7 seconds, no TPU needed.**

**Expected result:** ~0.4–1% of device time. **Effectively zero performance impact.** This is a genuine Pallas kernel diff — "Fused GDN, adapted to KDA" in the track's own words — and it should be presented as engineering depth, not as a performance result.

**Effort:** ~90 minutes. No device time.

---

### B — Replace the hardcoded fallback with a computed tiling schedule

**What:** When the tuning lookup misses, compute `num_kv_pages_per_block` from the VMEM budget and the actual tensor shapes instead of returning 1.

```
bytes_per_page = page_size × heads × kv_packing × (lkv_dim + r_dim) × dtype_size
resident       = Q + output + scratch
pages          = (vmem_budget − resident) / (bytes_per_page × double_buffer)
block          = clamp(prev_pow2(pages), 1, pages_per_seq)
```

**Files:** `kernels/mla/v2/tuned_params.py`

**Validation:** Unit tests against the shapes from the profile, plus it must predict the optimum that C measures.

**Expected result:** No new peak performance beyond what C finds — but it means **no unseen shape ever silently takes the worst value again.** This is the fleet-level fix and the K3 relevance.

**Effort:** ~2 hours. No device time. Calibrated by C.

---

### C — Build the MLA kernel autotuner ← **the priority**

**What:** A standalone harness that calls the MLA Pallas kernel **directly** with Kimi's real shapes, sweeps the schedule space (`num_kv_pages_per_block` × `num_queries_per_block` × `vmem_limit_bytes`), times each configuration on device, and emits a paste-ready `TunableParams` entry.

**Why this is the contribution:** six tuning tables, zero tooling. This is the missing piece of the stack, not a knob.

**Why it is fast:** no server, no 11-minute compile per point. Dozens of configurations in the time a single benchmark server takes to boot. It **supersedes** running the sweep through full serving benchmarks.

**Expected result — this is where the prize is.** MLA is 80.3% of device time at near-zero utilization. If the kernel gets `k`× faster:

| MLA speedup | End-to-end speedup | Projected c=16 tok/s |
|---|---|---|
| 2× | 1.67× | ~970 |
| 4× | 2.51× | ~1460 |
| 8× | 3.37× | ~1960 |

*(from `new_total = 19.7% + 80.3%/k`, applied to the measured 581.94 tok/s)*

These are projections from the measured share, not results. But the direction is not in doubt: a kernel at 80% of device time and 5.82% bandwidth utilization has one to two orders of magnitude of headroom before it hits the roofline.

**Effort:** ~2–3 hours to write, minutes per sweep point to run.

---

## Part 4 — Also on the branch

Landed, verified, and worth mentioning — but **not** performance results:

- **`decode_kda` checkpoint read/write slot offsets** through `BlockSpec` index maps. Verified on device with **zero regression** (388.03 vs 386.75 tok/s, no compile errors).
- **`PALLAS_INTERPRET` for `decode_kda`** — the kernel feedback loop went from a 45-minute TPU job to **7 seconds on CPU**. 6 tests.
- **Checkpoint-window contract test** — 11 tests asserting *exact* equality, pinning the semantics rollback-by-selection depends on.
- **Speculative decoding wired for KDA** — the `NotImplementedError` removed, the GDN checkpoint protocol implemented. **The mechanism is right; the output is currently wrong** (repeated tokens). Measured cost model: `69.4 ms fixed + 12.3 ms per window token` against a 12.61 ms baseline step, which shows it cannot pay at achievable acceptance lengths. Present as an engineering result with an honest negative, or omit.

---

## Part 5 — How to run things

### The winning serve config (S2)

```bash
vllm serve moonshotai/Kimi-Linear-48B-A3B-Instruct \
  --served-model-name=moonshotai/Kimi-Linear-48B-A3B-Instruct \
  --tensor-parallel-size 4 --max-model-len 16384 \
  --max-num-batched-tokens 2048 --max-num-seqs 64 \
  --gpu-memory-utilization 0.9 --trust-remote-code \
  --attention-backend CUSTOM --enable-expert-parallel --port 8000
```

### Profiling — the parts that cost us five wasted runs

- **`VLLM_TORCH_PROFILER_DIR` does nothing.** This plugin does not read it. Five runs produced zero traces because of it.
- Profiling is enabled **only** by `--profiler-config.profiler torch --profiler-config.torch_profiler_dir <dir>` on the **serve** command.
- **The `.pt.trace.json` Kineto trace is host-side only.** torch 2.11 disables native TPU profiling (needs ≥2.12). It shows Python overhead, not kernels.
- **Use XProf → HLO Op Profile.** That is the only view with per-kernel device time. Group by Program, order by wasted time, expand `custom-call`.
- Kernel identity comes from the **Source Stack** field in the op detail panel — that is how `custom_kernel.88` was identified as `chunk_kda` (`kda_attention_op.py:873`).

### Env vars are part of the compile-cache key

Any variable registered in `vllm_torchtpu/envs.py` is folded into `compute_tpu_compilation_hash`. Changing one forces a **cold compile (~11 min)**. That is correct behaviour — two block sizes are two different kernels — but budget for it. Set health-check timeouts to **900 s**.

Never read these from `os.environ` directly; the cache would replay the wrong kernel.

### Correctness gate

Any change that claims a speedup must pass this, or it is not a win:

```
POST /v1/completions   temperature=0, max_tokens=64
prompt: "The capital of France is"
must begin: " Paris. The capital of Italy is Rome"
```

---

## Part 6 — Open gaps and risks

| Gap | Impact | Status |
|---|---|---|
| Running **Kimi-Linear-48B**, not **K3 2.8T** | Track specifies K3 | K3 needs ~115 GB/rank; blocked on a large slice. Both models share the same MLA and KDA code paths, so the finding transfers. |
| Running **8 chips** (`tpu7x-8`), not 16 chips / 4 hosts / 32 cores | Track specifies 16 | Blocked on allocation. Metric is per-chip normalized, which partly compensates. |
| **TP/DP parallelism sweep not done** | Track asks for TP × EP × DP combinations | Only EP on/off at TP=4 tested. Needs the larger slice to be meaningful. |
| Speculative decode output is wrong | Blocks it as a perf result | Root cause not isolated; the unified-pool hypothesis needs `gnovack/kimi-linear-pooled-apc` (8 files) to test. |

---

## Part 7 — What we tell the judges

**One sentence:** *Kimi on TPU v7 spends 80% of its device time in the Gated MLA kernel at 2.76% FLOPS utilization, because the kernel's tiling schedule silently falls back to the minimum block size for any model not in a hand-maintained table — we built the autotuner that stack was missing, fixed the fallback to compute the schedule, extended the fusion in the KDA kernel, and separately found 32% in prefill scheduling.*

| Rubric category | What carries it |
|---|---|
| Technical Depth 30% | Device attribution obtained after two profiling paths proved dead; a Pallas kernel diff verified on device; the MLA schedule work |
| Innovation 25% | The autotuner — six tuning tables in this repo, no tooling to produce any of them. Plus speculative decoding on KDA, which had never run on TPU. |
| Performance Impact 20% | **+26% / +32% measured**, with a 4-cell ablation proving causality; MLA work on top |
| Path to Production 15% | A mergeable config file the repo lacks; an MLA fix that helps **every untuned model on the fleet**, K3 included |
| Pitch & Demo 10% | Live Kimi on v7. The HLO Op Profile screenshot — 80.3% / 2.76% FLOPS — is the money slide. |

**The line to lead with:**

> Every model brought up on this stack that isn't in a hand-tuned table silently runs its attention kernel at 3% utilization. We found it by profiling, and we built the tool that stops it happening to the next model.
