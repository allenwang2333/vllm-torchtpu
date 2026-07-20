# TPU Kernel Arena — Project Kickoff Design Doc

**Status:** Draft for formal kickoff
**Owner:** Qi Zhou (dreamingleo89@gmail.com)
**Prototype location:** `src/vllm_torchtpu/kernel_agents/`
**Last updated:** 2026-07-01

---

## 1. Executive Summary

We are formally kicking off **TPU Kernel Arena**, an adversarial multi-agent system that discovers, verifies, and ships hardware-optimal Pallas kernels for TPU inference. A working prototype exists in `vllm_torchtpu.kernel_agents`, has landed **two production-quality fused kernels** (rmsnorm_qkv_fuse: 1.055× TPOT geomean; rmsnorm_qkv_rope_fuse: 1.079× TPOT geomean) that have been numerically verified on real TPU v7x hardware, and demonstrates a **1.152× speedup at batch=64 on a real Qwen3-Coder-30B-A3B-Instruct pre-attention path**.

The gap this project closes: on GPU/CUDA, several published systems (INT21, CudaForge, Sakana AI CUDA Engineer, EvoEngineer, NVIDIA CUDA Agent) have demonstrated that adversarial LLM agents can outperform hand-tuned kernels. **No analogous system exists for TPU Pallas kernels.** The bottleneck on TPU kernel velocity today is the small pool of engineers who can hand-tune each new attention variant, MoE dispatch pattern, or quantization scheme. This project turns compute into kernel-engineer time — as our own model providers say, *"we can use compute to improve compute."*

Ask: **6-month funded initiative**, **2 FTE + 1 dedicated TPU v7 slice + ~$15K/month LLM inference budget**. Expected outcomes: 4-8 shipped fused kernels covering the attention, FFN, and quantization hot paths across Qwen, Llama, and DeepSeek families; a persistent skills / insights knowledge base that compounds across runs; a reusable framework other TPU-serving stacks (SGLang, JetStream) can consume.

---

## 2. Motivation

### 2.1 The TPU kernel velocity bottleneck

Every new hardware generation (v5e → v6e → v7 → v8), every new model architecture (dense → GQA → MoE → MLA → hybrid), and every new quantization scheme (int8 → fp8 → mxfp4 → int4) requires re-tuning the kernel library. On TPU this means new Pallas kernels with new block sizes, new memory hierarchies (VMEM budgets, SMEM tiling), new dispatch patterns.

Today this work is done by a small pool of engineers with deep Pallas expertise. Bottleneck symptoms:

- **New model launch lag**: 2-6 weeks between HF checkpoint drop and shippable TPU kernel path for hot ops.
- **Perf debt**: kernels tuned for v6e are still running on v7 because nobody has retuned block sizes. Live example in `src/vllm_torchtpu/kernels/quantized_matmul/tuned_block_sizes.py` — v6 entries only, v7 falls back to `(128, 128, 128)` which is measurably suboptimal on Qwen 30B shapes.
- **Missed fusions**: the XLA compiler doesn't fuse across Pallas call boundaries. Hand-writing every fused variant (RMSNorm+QKV, RMSNorm+QKV+RoPE, residual+RMSNorm+QKV, SwiGLU FFN, ...) is O(N²) manual work.

### 2.2 The turning point

Three developments converged in 2025-2026:

1. **LLM code generation is now good enough** to write correct Pallas kernels one-shot, with structured prompting. Empirically confirmed in this prototype: Anthropic Claude Opus 4.7 and OpenAI GPT-5.5 both produce compilable, numerically correct Pallas kernels from a task spec + persona hint.
2. **Adversarial workflows unlock quality**. INT21 (Bing Xu, MSFT), NVIDIA CUDA Agent, and CudaForge have shown that pairing a Coder LLM against a Critic/Judge LLM with hardware-verified feedback consistently outperforms single-LLM prompting by 15-30 percentage points.
3. **Reasoning models change the economics**. `claude-opus-4.7` and `gpt-5.5` cost ~$50/campaign vs. ~$25K/eng-week for a hand-tuned kernel. **The break-even is a single winning campaign per week.**

### 2.3 What this project delivers that a solo engineer cannot

- **24/7 exploration**. The arena runs continuously against a task queue.
- **Systematic negative results**. When a fusion pattern *cannot* beat XLA on workload, the arena reports it with hardware evidence. Engineers today have no equivalent "we tried and it doesn't work" record.
- **Provenance and reproducibility**. Every candidate is tied to (task spec, generation, persona, LLM backend, prompt hash). Every winner has an autopsy trail.
- **Compound value across runs**. Insights ("Pallas call breaks XLA cross-op fusion — small outputs win"; "int8→bf16 in-kernel dequant beats separate dequant kernel") accumulate in a persistent KB and warm-start future runs on same-family tasks.

---

## 3. Related Work

| System | Target | Contribution the prototype builds on |
|---|---|---|
| INT21 (Bing Xu, 2025) | CUDA kernels | Adversarial Author+Critic loop pattern with hardware verification |
| NVIDIA CUDA Agent | CUDA | RL-trained agent reaches 96.8% "faster-than-torch.compile" from 66% prompted baseline; establishes value of RL trajectories |
| CudaForge | CUDA | Two-mode Judge (numerics + hardware); curated NCU metrics feedback to Coder |
| Sakana AI CUDA Engineer | CUDA | Persona-seeded parallel Authors; elite propagation across generations |
| EvoEngineer | CUDA | Elite + insight pool propagation beats homogeneous mutation |
| AccelOpt | GPU | Task synthesis for training data at scale |
| Astra | GPU | Speed-of-Light DSL guidance for LLMs |

**The gap:** every published system targets NVIDIA CUDA. **TPU Pallas has no analogous adversarial arena.** This project takes the design patterns that work on CUDA, adapts them to TPU's very different execution model (MXU + VPU + scalar core, VMEM hierarchy, per-Pallas-call XLA fusion boundary), and ships production kernels into vLLM.

---

## 4. System Design

### 4.1 Architecture

```
                            ┌──────────────┐
                            │ Task registry│  ← rmsnorm_qkv, softmax, quantized_matmul,
                            └──────┬───────┘    swiglu, ..., + task synthesizer
                                   │
                                   ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                        Arena Loop                            │
    │                                                              │
    │   [Author×N]  → [Verifier] → [Timer] → [Scorer]              │
    │       │             (numerics)   (hardware)   (workload²)    │
    │       │                                          │           │
    │       ▼                                          ▼           │
    │   [Personas]                             [Elite pool]        │
    │       │                                          │           │
    │       │            ┌────────────┐                │           │
    │       ├────────────│ Insights KB│◄───────────────┤           │
    │       │            └────────────┘                │           │
    │       ▼                                          ▼           │
    │   [Critic] ← rooflines, XPlane traces, HLO diff              │
    │       │                                                      │
    │       ▼                                                      │
    │   [Optimizer×K] → [Reviewer (adversarial gate)] → next-gen   │
    └──────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                            ┌──────────────┐
                            │  Ship CLI    │  → branch + kernel replacement + PR
                            └──────────────┘
```

### 4.2 Key components (with file references)

| Component | File | Purpose |
|---|---|---|
| `KernelTask` | `core/task.py` | Task spec: reference fn, baseline fn, shape grid, tolerance |
| `WorkerPool` | `core/sandbox.py` | Long-lived subprocess/in-process workers; AST scrub, timeout, recycle |
| `Verifier` | `core/verifier.py` | Numerics gate at task's dtype tolerance |
| `Timer` | `core/timer.py` | Warmup+median+outlier reject |
| `Scorer` | `core/scorer.py` | `combined_score = kernel_geomean × workload_geomean²`; ship gate at 0.98 per-workload TPOT |
| `Arena` | `core/arena.py` | The orchestrator — persona-seeded, insights-propagating |
| `Author`, `Critic`, `Optimizer`, `Reviewer` | `agents/` | LLM roles |
| `AnthropicBackend`, `VertexBackend`, `OpenAIBackend`, `GeminiBackend`, `MockBackend` | `backends/` | Provider-agnostic dispatch |
| `FrameworkAdapter` | `framework_adapters/` | vLLM adapter shipped, SGLang stubbed |
| `InsightsDB` | `core/insights_db.py` | Cross-run persistent KB, task-family × TPU-generation keyed |
| `AntiHacking` | `core/anti_hacking.py` | AST scrub for reward-hack patterns + HLO-diff structural check |
| `Roofline`, `Profiler`, `XPlane` | `core/roofline.py`, `profiler.py`, `xplane.py` | Speed-of-Light guidance + barrier/stall attribution |
| `RLTrajectoryLogger` | `core/rl_trajectory.py` | Append-only JSONL for downstream RL training |
| `TaskSynthesis` | `core/task_synthesis.py` | Combinatorial op-family enumeration for bulk task generation |
| `Ship CLI` | `ship.py` | Branch creation + kernel replacement + inferencex report |

---

## 5. Key Design Choices, with Rationale

Each choice below was made after considering at least one alternative. The rationale explains **why** — this is the "defend the design in review" section.

### 5.1 Workload-primary scoring (`combined_score = kernel_geomean × workload_geomean²`)

**Alternative:** rank by kernel-only speedup (what most published arenas do).

**Why we chose workload-primary:** every `pl.pallas_call` is an XLA fusion boundary. A kernel-only benchmark measures the isolated op against XLA's fused-to-neighbors baseline; but in a real transformer-layer chain, the baseline gets fused across many layers while a Pallas call breaks that fusion. Empirically confirmed in this prototype: **three arena campaigns (residual_rmsnorm_qkv_fuse, rmsnorm_gate_up_fuse, swiglu_ffn_fuse)** produced candidates that hit 1.06×–1.13× on isolated-kernel timing but *lost* on real-workload TPOT chains (0.87×–0.90×). A kernel-only ranking would have shipped three regressions. **Workload-primary scoring is what separates a serious kernel arena from a benchmark-gaming tool.**

### 5.2 Persona-seeded Authors, neutral Optimizers

**Alternative:** homogeneous LLM prompting for all agents (INT21's original approach).

**Why:** persona bias in Authors (memory-bound / MXU-thrasher / minimalist / spill-hunter / fusion-enthusiast) produces a diverse initial population — no single "blind spot" dominates. Optimizers are neutral so they can accept the best insight from any persona's output. This mirrors EvoEngineer's finding that elite + insights beats persona-locked mutation.

### 5.3 Insights pool that propagates across generations (and runs)

**Alternative:** stateless generations, each starting fresh.

**Why:** the same insight ("VMEM scratch reuse across the n-loop saves one HBM re-read per output tile") applies to every 2-op fusion. Emitting it once and reusing it across generations shortcuts what would otherwise be N-generation re-derivations. Cross-run persistence (task-family × TPU-generation) means the arena warm-starts a new run on `rmsnorm_qkv_rope_fuse` with the insights that won on `rmsnorm_qkv_fuse` — validated: this cut Optimizer discovery time for the rope variant to 3 generations instead of the 8 that seeded runs typically take.

### 5.4 Meta-adversarial Reviewer with per-shape regression prediction

**Alternative:** always evaluate every Optimizer output on TPU.

**Why:** TPU time is the scarcest resource in the loop. A Reviewer that predicts per-shape regressions from source diff + parent per-shape numbers rejects ~15-25% of Optimizer output before it burns TPU cycles. Empirically we hit a bug where the Reviewer was *too* strict (blocked every mutation that wasn't a pure tuning tweak); the fix was a composite `should_block` gate requiring EITHER a geomean-below-parent prediction OR two-or-more predicted shape regressions. Now the Reviewer catches obvious losses without over-blocking exploration.

### 5.5 Provider-agnostic backend + framework-agnostic adapter

**Alternative:** hard-code Anthropic + vLLM.

**Why:** an industry-leading TPU kernel arena needs to be adoptable by teams that don't use Anthropic (they use Gemini, GPT, or a local endpoint) and don't use vLLM (they use SGLang, JetStream, or a custom stack). Shipping six providers (Anthropic direct/Vertex, OpenAI direct/compatible, Gemini AI Studio/Vertex, Mock) at day one dramatically widens the adoption surface. The framework-adapter Protocol was proven with the vLLM adapter and stubbed for SGLang; a JetStream adapter is ~200 LOC away.

### 5.6 Persistent insights KB + Skills directory

**Alternative:** monolithic system prompt with everything.

**Why:** system prompts get too long, degrade quality, and can't be updated without an SDK release. Skills are Markdown files loaded on demand by task family (`skills/pallas_xla_boundary.md`, `skills/tpu_pallas_apis.md`, `skills/winning_fusion_template.md`). Insights are shorter, per-run, and versioned in a JSON KB. This matches the pattern NVIDIA's CUDA Agent RL work used with SKILL.md — the arena benefits from documented, task-family-scoped guidance without prompt bloat.

### 5.7 Anti-hacking: AST scrub + HLO-diff structural check

**Alternative:** trust the LLM.

**Why:** LLMs will reward-hack. Observed patterns:
- Candidate imports and calls the baseline it's supposed to beat (`xla_quantized_matmul`).
- Candidate returns `jnp.zeros_like(x)` if tolerance is loose.
- Candidate JIT-wraps the reference expression verbatim.

AST scrub catches the first two. HLO diff catches the third (candidate compiles to structurally identical HLO as baseline). Both are advisory to the Reviewer, not automatic rejections — because the workload gate is authoritative, and a "trivial wrapper" that happens to win *is* still a valid finding.

**Note:** HLO-diff check is kept as an offline diagnostic in the current code. The check `exec`s untrusted source, so it must run inside a WorkerPool worker (with timeout + recycle) before it can be wired into the arena's per-candidate hot path. Followup work.

### 5.8 Roofline + XPlane feedback to LLMs

**Alternative:** give LLMs only speedup numbers.

**Why:** speedup alone doesn't tell the LLM *why* — is the kernel HBM-bound, MXU-bound, or barrier-stalled? Roofline analysis (achieved TFLOPS / peak, achieved GiB/s / peak) plus XPlane barrier/stall attribution (DMA wait %, sync wait %, MXU active %) let the LLM reason about the right axis to attack next. In one measured run this shortened iteration from 8 generations to 4 to reach a plateau, because the Optimizer stopped exploring compute-side changes when the profiler said the kernel was memory-bound.

### 5.9 RL trajectory logger with tri-state `verifier_ok`

**Alternative:** just log (state, action, reward) tuples.

**Why:** NVIDIA's CUDA Agent RL work went from 66% → 96.8% "faster-than-torch.compile" via RL training on trajectories. To make the arena's data RL-ready, three states must be distinguishable: numerics failed (`verifier_ok=False`), reviewer blocked without running (`verifier_ok=None`), verified pass (`verifier_ok=True`). A trainer that recomputes reward from a binary field would confuse "never ran" with "ran and failed" and produce a bad policy.

### 5.10 Task synthesis engine for RL training data at scale

**Alternative:** hand-curate every task.

**Why:** RL training needs *thousands* of tasks, not the 16 shipped ones. The `synth` CLI combinatorially enumerates 2-3-op chains from a fixed op family (rmsnorm, softmax, gelu, silu_mul, residual_add, matmul), deduplicates, filters by baseline runtime, and emits an importable module. This scales the arena's training-data surface from human bandwidth (dozens/quarter) to compute bandwidth (thousands/day).

---

## 6. Real Results (current prototype)

### 6.1 Framework state (verified)

- **129 unit + integration tests passing** in `tests/kernel_agents/` (~52s on TPU v7x).
- **Six providers** shipped: Anthropic direct, Anthropic Vertex, OpenAI direct, OpenAI-compatible (Groq/Together/OpenRouter/local vLLM/LM Studio), Gemini AI Studio, Gemini Vertex, plus Mock for CI.
- **Sixteen registered tasks** covering RMSNorm, softmax, matmul, quantized matmul, SwiGLU FFN, various fusion variants.

### 6.2 Shipped kernels (arena-discovered, hardware-verified)

| Kernel | Arena speedup (TPOT geomean) | Correctness verified? | Discovery cost |
|---|---|---|---|
| `rmsnorm_qkv_fuse` | **1.055×** (1.084× / 1.032× / 1.054× per shape) | ✅ TPU v7x, bf16 tolerance | 1 Vertex campaign, ~25 min, ~$8 |
| `rmsnorm_qkv_rope_fuse` | **1.079×** (1.038× / 1.121× per shape) | ✅ TPU v7x, bf16 tolerance | 1 Vertex campaign, ~30 min, ~$10 |

### 6.3 Real-workload benchmark (Option B, `benchmarks/bench_rmsnorm_qkv_fuse_qwen3.py`)

**Setup:** Qwen3-Coder-30B-A3B-Instruct shapes (hidden=2048, num_q_heads=16, num_kv_heads=2, head_dim=128), INT8 weights, TPU v7x8. Fused kernel vs unfused (RMSNorm + xla_quantized_matmul), both include q_norm/k_norm/RoPE as separate ops after the fused kernel — matching Qwen3's actual pre-attention architecture.

| Batch | Unfused (μs) | Fused (μs) | Speedup |
|---:|---:|---:|---:|
| 64 | 294.6 | 255.8 | **1.152×** |
| 128 | 241.5 | 238.6 | 1.012× |
| 256 | 241.7 | 242.7 | 0.996× |

**Geomean 1.051×.** Numerical equivalence: max_abs_diff ≤ 0.016, rel_err ≤ 0.15%. The arena's synthetic-workload 1.055× translates almost 1:1 to a real Qwen3-shape scenario.

### 6.4 Negative results (arena working as designed)

The arena correctly refused to promote three candidates in this session that had kernel-only wins but failed the workload gate:

| Task | Kernel-only best | Workload best | Arena verdict |
|---|---:|---:|---|
| `residual_rmsnorm_qkv_fuse` | 1.06× | 0.87× | Not shipped — 3-op fusion loses to XLA cross-layer fusion |
| `rmsnorm_gate_up_fuse` | ~1.00× | 0.87× | Not shipped — pattern doesn't generalize to 2H output |
| `swiglu_ffn_fuse` | 1.13× | 0.90× | Not shipped — same 3-op failure mode |

**A kernel-only ranking would have shipped three regressions.** This is the arena's core value proposition demonstrated.

---

## 7. Impact and Business Case

### 7.1 Direct kernel-perf value

At **1.05× TPOT** on the pre-attention path across all 48 Qwen 30B decoder layers, an end-to-end integration would yield **~2-3% TPOT** — measurable, defensible on the leaderboard. Compound across attention, FFN, and sampler wins reasonably projected from a 6-month campaign: **5-10% end-to-end TPOT** on the main serving model.

At the current TPU v7 rate card, a 5-10% inference speedup on a production 30B-scale model directly translates to **5-10% cost reduction** on that serving fleet, or **5-10% headroom for larger models at fixed cost**.

### 7.2 Time-to-market value

New model families (DeepSeek V3, Qwen3-MoE, Llama 4 hybrid) currently take **2-6 weeks** from HF drop to shippable TPU kernel path. With an operating arena:

- **First working kernel** in hours (one Author round).
- **Optimized kernel** in a day (one full campaign).
- **Integration** — separately tracked.

Halving new-model-launch lag is a direct product value for the TPU Inference org.

### 7.3 Retention of institutional knowledge

The insights KB accumulates task-family × TPU-generation guidance across every run — "VMEM scratch reuse across the n-loop wins for 2-op fusions with dominant matmul" is now written down, not tribal knowledge in three engineers' heads. When those engineers rotate, the KB retains their signal.

### 7.4 Reusable adversarial workflow

Beyond kernels, the *framework* is reusable. Any TPU-hot path where hand-tuning is the bottleneck (custom collectives, compilation passes, sampler kernels) can be re-cast as a `KernelTask` and driven by the same arena. This is a general capability, not a one-shot kernel-shop.

---

## 8. Resource Allocation

### 8.1 Team

| Role | Count | Rationale |
|---|---|---|
| Kernel engineer (framework lead) | 1 | Owns arena core, ship pipeline, and integration with vllm_torchtpu |
| Kernel engineer (kernel & integration) | 1 | Runs campaigns, wires winners into serving stacks, handles model-registry integrations |
| LLM/prompt engineer (part-time, 40%) | 0.4 | Owns skills directory, prompt evolution, backend integrations, agent-persona tuning |
| Eng manager oversight (10%) | 0.1 | Prioritization, resource negotiation |

**Total: 2.5 FTE across the 6-month kickoff phase.**

### 8.2 Compute budget

| Resource | Estimate | Notes |
|---|---|---|
| TPU v7 slice (dedicated) | 1 × v7x8 for the 6 months | Shared with adjacent perf work is acceptable; dedicated preferred |
| LLM inference (Vertex Anthropic + OpenAI + Gemini) | **~$15K/month** ($90K over 6mo) | Rough model: 3 campaigns/week × $30/campaign for author+critic+optimizer+reviewer rounds × safety margin for exploration. Actual measured: ~$8-10/campaign in this prototype. |
| Vertex + storage (arena archive) | ~$500/month | JSONL archives, run artifacts, KB storage |
| Human eng time | 2.5 FTE × 6 months | ~$400K fully loaded |

**Total 6-month cost: ~$500K.**

### 8.3 Expected ROI

At even a **conservative 3% end-to-end TPOT improvement** shipped to the primary serving fleet over 6 months, plus 2-3 weeks average reduction in new-model launch lag, the initiative is comfortably ROI-positive on inference cost savings alone. The framework's ongoing value (compounding wins, retained institutional knowledge, adaptation to new hardware generations) is upside on top.

---

## 9. Execution Roadmap

### Phase 1 (Months 0-2): Foundation solidification

- **P0:** wire `rmsnorm_qkv_fuse` and `rmsnorm_qkv_rope_fuse` into `vllm_torchtpu` model builders for Qwen2-family models (kernels directly apply); measure end-to-end TPOT on Qwen2-72B (or Qwen2.5-7B if 72B unavailable) and land the first measured end-to-end serving win.
- **P0:** move HLO-diff anti-hacking into the WorkerPool sandbox so it can run per-candidate.
- **P1:** finalize the `synth` → arena pipeline: emit synth tasks → run arena → collect trajectories → feed to (external) RL training.
- **P1:** ship the SGLang framework adapter (currently stubbed).
- **P2:** open source the framework under `vllm_torchtpu.kernel_agents` with the current shipped kernels as reference wins.

### Phase 2 (Months 2-4): Kernel expansion

- Target the FFN path (SwiGLU / down_proj variants) with the workload gate as an honest referee. Prior campaigns show 3-op FFN fusions don't win on workload; the follow-up is to find 2-op fusions within FFN that do — e.g. `gate_up_proj + silu_mul` alone.
- Target attention hot paths: fused RPA v3 variants, prefill/decode variants of quantized matmul, MoE dispatch.
- Target quantization: v7-native fp8 kernels (currently v6 tuned tables fall back to naive on v7).

### Phase 3 (Months 4-6): Automation

- **Auto-PR pipeline**: when an arena run produces a workload-gated winner, `ship` CLI opens a PR against `vllm_torchtpu` main with the kernel + call-site wire-up + benchmark result. Human review remains gating.
- **CI integration**: nightly arena runs on the tuned tasks with regression alerts if a shipped kernel's speedup shrinks.
- **Multi-device kernels**: extend the sandbox to support kernels that call `torch.distributed` collectives (currently single-chip only).

### Phase 4 (Months 6-12, if funded): RL-trained agents

- Use the accumulated trajectory data to train a Coder LLM that outperforms prompted Claude Opus on Pallas kernels specifically (NVIDIA CUDA Agent showed 66% → 96.8% jump).
- Deploy the RL-trained agent as a first-class backend (`--backend rl_agent`).

---

## 10. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| LLM providers change pricing / rate limits mid-project | Medium | Medium | Six providers already integrated; automatic failover via `--backend` |
| A generation of TPU (v8) invalidates tuned block sizes | High | Medium | Insights KB is TPU-generation keyed; runs re-tune automatically on new hardware |
| XLA compiler improvements close the fusion gap we exploit | Medium | High | Workload-primary scoring surfaces this immediately; arena stops promoting patterns that no longer win |
| Winners look great in arena but don't survive real production traffic | Medium | High | Ship gate at per-workload TPOT ≥ 0.98 already forces workload realism; Phase 1 end-to-end integration is the final defense |
| Cost overrun on LLM inference | Medium | Low | Budget throttle via `budget.total` in workflow; workflow tokens counted per run |
| Team member turnover mid-project | Medium | Medium | Skills directory + insights KB retain knowledge outside individuals |

---

## 11. Success Metrics

**Six-month scorecard (targets to hit for renewal):**

- ≥ **4 kernels shipped** to production Qwen and Llama serving paths with end-to-end TPOT wins ≥ 1.02× measured under load.
- ≥ **10% reduction** in new-model launch lag (measured from HF drop to first working kernel on TPU).
- ≥ **200 tasks** in the persistent registry (mostly synth-generated).
- ≥ **500 arena runs** completed with hardware-attributed results.
- **Zero regressions** shipped (workload gate not overridden).
- **SGLang adapter** shipped, second serving framework consuming winners.

**Twelve-month scorecard (renewal):**

- ≥ **5% end-to-end TPOT win** on the primary Qwen-family serving fleet.
- RL-trained Coder agent operational, beating prompted Claude Opus by ≥ 10 percentage points on new task-family generalization.

---

## 12. Open Questions

1. **Deployment governance:** who reviews auto-PRs from the arena? Proposed: rotate perf-eng oncall, single approver, `ship` output must include the workload-gate report + at-least-one attached run archive JSON.
2. **RL training compute:** phase 4 needs training-cluster time. Not budgeted here; would be a separate ask if / when phase 3 is complete.
3. **Multi-tenant TPU:** can the arena run alongside production serving on a shared TPU? Prototype currently locks the whole chip. Multi-tenant would require queue arbitration.
4. **Provider spend cap:** at what monthly $ do we throttle exploration? Proposed: soft cap at $15K/month, hard cap at $25K.

---

## Appendix A. Concrete file references

- Framework entry: `src/vllm_torchtpu/kernel_agents/__main__.py`
- Ship CLI: `python -m vllm_torchtpu.kernel_agents ship <task> <run_id>`
- Task list: `python -m vllm_torchtpu.kernel_agents list`
- Insights KB browse: `python -m vllm_torchtpu.kernel_agents kb --task <family>`
- Framework tests: `pytest tests/kernel_agents/ -q` — 129 passed
- Kernel correctness tests (real TPU): `pytest tests/kernels/test_rmsnorm_qkv_fuse_kernel.py`, `test_rmsnorm_qkv_rope_fuse_kernel.py`
- End-to-end Qwen3-shape benchmark: `python benchmarks/bench_rmsnorm_qkv_fuse_qwen3.py`

## Appendix B. Branch structure (current)

| Branch | Head | Contents |
|---|---|---|
| `kernel-agents/framework` | 20eba3d4 | Full framework, tests, docs, archived winners |
| `kernel-agents/rmsnorm-qkv-fuse` | 18a45c8e | Kernel module + torch op + correctness test + Qwen3 benchmark |
| `kernel-agents/rmsnorm-qkv-rope-fuse` | 7999a4e4 | Kernel module + torch op + correctness test |

## Appendix C. Prior art references

- Bing Xu (MSFT) — INT21 adversarial CUDA kernel work
- NVIDIA CUDA Agent RL training + SKILL.md pattern
- CudaForge — two-mode Judge, curated NCU metrics
- EvoEngineer — elite + insight propagation over LLM textual crossover
- Sakana AI CUDA Engineer — persona-seeded parallel Authors
- Astra — Speed-of-Light DSL guidance
- AccelOpt — combinatorial task synthesis at scale
