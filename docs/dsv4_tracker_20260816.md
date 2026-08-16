# DeepSeek-V4: everything in flight

> **Written 2026-08-16, last updated 19:41 UTC.** This file is the source of
> truth for DSv4 as of that timestamp.
>
> **Supersedes and replaces** the earlier handoff docs, which were deleted on
> 2026-08-16 because they contradicted each other: `dsv4_remaining_work.md`,
> `dsv4_next_session_prompt.md`, `handoff_refactor.md`, `possible_prompt.md`,
> `pr_review_tracking.md`, `docs/dsv4/`, and
> `../scratchpad/HANDOFF_20260814_evening.md`.
>
> `dsv4_handoff_20260806.md` survives only because `quant_weight_cache.py:28`
> cites its section 7c; it is historical and where it disagrees with this file,
> it is wrong. Several older docs record claims disproved on 2026-08-16 — see
> the Refuted table.
>
> **Companion:** `dsv4_next_session_20260816.md` holds the pickup prompt.

Nothing counts as finished until it is **written**, **verified**, **pushed** and
**merged**, in that order. The status column names the last stage an item has
actually reached, so an item sitting at *written* has been edited and statically
checked and nothing more. Most of this branch is at *written*. Other statuses:
*running*, *blocked* (waiting on a named item), *open* (no work started),
*deferred*, *overdue* (a commitment made to a reviewer and not met).

Priority is what to spend the next TPU run on, not what is most interesting.

**Current objective (Pate, 2026-08-16):** accuracy parity with tpu-inference's
DSv4, expressed in the structure `kimi_k3` already uses in this repo. *Not*
exact parity with tpu-inference, and not its memory or compile characteristics.
Accuracy parity is met (A6). The remaining work is design parity — the G rows.
The compile-ceiling question is closed: A8 was refuted on 2026-08-16 and
compiled serving works at 9216 (A14).

## Picking this up cold

Everything below is on host `alynie-v7x-8`, a single Google Cloud TPU v7x-8:
**eight chips, one tenant at a time**. There is no second machine.

| What | Where |
|---|---|
| Repo, working branch | `~/workspace/vllm-torchtpu`, branch `patemotter/dsv4-native-model` |
| Reference implementation | `~/workspace/tpu-inference` (pinned `f329f3480`) |
| Python env for everything | `~/workspace/uv_venv_dsv4/bin/python` |
| Model weights | `~/workspace/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1` |
| Quantized-weight cache | `~/workspace/dsv4_quant_cache` (132 GB, on disk — **never** `/dev/shm`) |
| Run logs, gate scripts | `~/workspace/vllm-torchtpu/scratchpad/` (untracked; not in git) |
| Historical eval artifacts | `~/workspace/eval_results/` (untracked) |
| `ruff` | not installed; use `uvx ruff@latest` |

Read `scratchpad/chain_progress.log` first — it is the append-only record of
every run, in order, with verdicts.

## File map

Model, in dependency order:

- `src/vllm_torchtpu/models/vllm/deepseek_v4/model.py` — the whole model:
  `DeepseekV4MLP`, `MoE`, `DecoderLayer`, `Head`, `Model`, `ForCausalLM`, the
  weights mapper, and the `mhc_*` helpers. 30 KB, not yet split (see G3).
- `src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4/` — attention,
  compressor, indexer. Note this is a *different tree* from the model (G4).
- `src/vllm_torchtpu/layers/vllm/quantization/deepseek_v4_fp8.py` — the
  `deepseek_v4_fp8` quant config; dispatches linear vs MoE methods.
- `src/vllm_torchtpu/layers/vllm/quantization/fp8.py` — `VllmFp8LinearMethodTPU`
  and `VllmFp8MoEMethodTPU`. Shared with every other FP8 model, so DSv4-specific
  behavior here needs justifying.
- `src/vllm_torchtpu/kernels/deepseek_v4/` — Pallas kernels: `mla`, `mla_swa`,
  `compressor`, `compress_norm_rope`, `streamindex_topk`. **Trusted by default**
  per the working agreement; suspect our code first.

Tests: `tests/kernels/test_dsv4_*`, `tests/layers/vllm/test_dsv4_*`,
`tests/models/vllm/test_dsv4_model.py`.

Gate prompts: `tools/dsv4_debug/prompts/` — a 10-prompt `pool/`, plus
`needle_250.txt` (the in-window recall control, answer `7391-QUARTZ-4408`),
`short.txt`, `xla_llvm_prompt.txt`.

Baselines to diff against: `scratchpad/gate_cache_compile.sub` is the current
compiled reference. A `.sub` file is the extracted substantive lines from a
`.log`; diff the `Generated text:` and `Generated token IDs:` lines.

## Running things

```bash
# Compiled gate — the standard correctness check. ~30 min. 1024 context.
setsid nohup scratchpad/gate_dsv4_compile.sh . <tag> > scratchpad/<tag>_wrapper.log 2>&1 &
# Pass: tracebacks=0 sweeps=13 raises=0, needle exact, generations match baseline.

# Eager gate — same but VLLM_ENFORCE_EAGER=1, 26 sweeps. Slower; use only to
# isolate a compile-specific suspicion. Serving should always be compiled (A13).
setsid nohup scratchpad/gate_dsv4.sh . <tag> > ... &

# MMLU-Pro accuracy, ~100 min at limit 20. Server-based.
setsid nohup scratchpad/run_dsv4_mmlupro.sh > scratchpad/mmlupro_wrapper.log 2>&1 &

# Scale-equivalence proof, CPU only, safe while a gate runs.
JAX_PLATFORMS=cpu ~/workspace/uv_venv_dsv4/bin/python scratchpad/compare_scale_equivalence.py
```

Name the tag after the commit being gated, and commit before gating — see the
provenance warning under Run log for why.

## Operating the TPU — hazards that cost time

**Is the TPU free?** The only check that works:

```bash
for p in /proc/[0-9]*; do ls -l $p/fd 2>/dev/null | grep -q vfio && \
  echo "$p: $(tr '\0' ' ' < $p/cmdline | cut -c1-80)"; done
```

`lsof /dev/vfio/*` reports zero holders while a server is running — it is
useless here. `pgrep -f entrypoints.openai.api_server` also misses, because vLLM
renames its workers to `VLLM::Worker_TP0_EP0` and orphans keep no trace of the
parent command line. Both of those misled this session repeatedly.

**Cleaning up after a kill.** `pkill -9 -f "VLLM::"`, then remove
`/dev/shm/psm_*` and `/tmp/libtpu_lockfile*`. `run_dsv4_mmlupro.sh` does this in a trap, and `gate_dsv4_compile.sh` now does
it at startup too (added 2026-08-16 after it cost two gate attempts). Without
it a hard-killed run poisons the next gate with `PjRtClient is not initialized`
/ `Device or resource busy; Couldn't open iommu group` — an error that looks
like a code bug and is not one.

**Never run `pytest` while anything is serving** — it takes the chips.

**`LOAD_FORMAT=dummy`** is valid for shape, sizing and compile questions only.
Never for anything depending on weight values.

`/dev/shm/torch_tpu_cache` currently holds 109 GB of RAM. Not urgent, but it is
the same category of problem as the weight cache that was moved to disk.

**Utilization.** Every other model on this hardware runs `gpu-memory-utilization`
at 0.92–0.95; every working DSv4 config sits at 0.48–0.55. That is not
conservatism — at 0.92 the DSv4 KV init fails asking for 2.19 GB with 1.60 GB
free. See G5. Until that is fixed, treat high utilization as unavailable for
DSv4 and record the highest value observed to work.

**Use `scripts/vllm/benchmarking/cleanup_server.sh`** to free the chips. It
already handles `vllm serve`, `vllm.entrypoints`, `VLLM::` workers and
`/dev/vfio`, and it succeeded where an ad-hoc `pkill` left orphans behind.

## Scope

**In scope now:** design parity — the G rows. Accuracy parity is met and needs
no further work unless something changes.

**Out of scope, deliberately:** exact parity with tpu-inference; its memory
characteristics; DP attention (E2); the perf gap vs tpu-inference (E1). None of
these block the current objective. The compile ceiling is no longer out of
scope because it is no longer a question — see A14.

**Never done by the agent:** opening PRs, posting review comments, merging.
Draft and hand over.

| # | Item | Status | Pri | Evidence / reference | Next step |
|---|---|---|---|---|---|
| A1 | MMLU-Pro 0.7786 ± 0.0154 over 700 samples, per-subject spread from law 0.54 to math 0.92 | verified | — | `eval_results/phase3_limit50/.../results_2026-08-12T15-49-05.json`, `limit: 50`, filter `custom-extract` | Nothing; this is the working-model reference point |
| A2 | MMLU (`mmlu_llama`) 0.2379 mean over 62 subtasks against 0.25 chance, no subtask above 0.347 | explained by A4 | P3 | `scratchpad/mmlu_results/*/results_2026-08-15T12-54-29*.json`, full run, `limit: None` | Treat this score as meaningless until the serving path is fixed |
| A3 | `mmlu_llama` prompts end with a partial *assistant* turn, `"The best answer is"`, and `continue_final_message: true`; the model is meant to continue it with one letter and stop at `.` | verified | — | `samples_mmlu_llama_astronomy_2026-08-16T02-37-45.jsonl`, `arguments.arg_0` | Reference for A6 |
| A4 | The model does not continue it: **50 of 60 responses are the literal letter `C`** — zero A, B or D — and the other 10 open a fresh `# Explanation` block | verified | — | same file, `resps`; run `scratchpad/run_dsv4_mmlu_samples.sh` 02:24–02:37 | Constant `C` scores ~0.25 by accident; the 10 explanations score 0. That is the whole of A2 |
| A5 | Inferred mechanism: `continue_final_message` is not stitched through the `local-chat-completions` → vLLM chat path, so the model answers a prompt that already ended | inferred | P3 | 50/50 identical output is degenerate even for a mis-primed prompt, so the mechanism is still unproven — but A6 rules out the model, so the fault is in that path | Only worth chasing if we need `mmlu_llama` specifically; MMLU-Pro is the working harness |
| A6 | MMLU-Pro on this branch: **0.7893 ± 0.0241** over 280 samples vs the baseline's 0.7786 ± 0.0154 — 0.37σ apart, no regression | verified | — | `scratchpad/mmlupro_results/.../results_2026-08-16T08-09-35.json`, `limit 20`, eager. Per-subject shape matches: physics/biology 0.95, law/history/philosophy 0.65 | Closes A2, A3, A5. The model is fine; `mmlu_llama`'s serving path is the only broken thing |
| A7 | MMLU-Pro runner: `lm_eval` flags from `scratch/run_mmlu_limit100.sh`, server and env from `gate_dsv4_compile.sh` | written | — | the four `scratch/run_mmlu*.sh` all export `LIBTPU_INIT_ARGS` and `MOE_REQUANTIZE_*`, the env behind the repeating-output harness bug, so they are not safe to run verbatim | Reuse `scratchpad/run_dsv4_mmlupro.sh` for future MMLU-Pro runs |
| A8 | ~~Compiled serving fails at 9216~~ — **REFUTED** | **refuted 2026-08-16** | — | `PASS 9216 18:52:02` on `d4f8c153`: compiled, KV 308,292 tokens, 33.45× concurrency, served a request. The 103.09 GB failure in run-log row C happened only on a tree with `sys.setrecursionlimit(50000)` removed — a defect I introduced and have since reverted | Nothing. There is no compile ceiling at production context |
| A11 | ~~`de8a5a1c` ships a regression~~ — **moot**, A8 refuted | closed | — | compiled serves at 9216 with that commit in place. The reference compiles its backbone as one graph too | Nothing. D1 is not blocked by the compile question |
| A12 | gxd3 agreed to the refactor "after PR11", but PR11 was never opened | dropped | — | no kv-admission PR in `gh pr list --author @me --state all`. Pate's call 2026-08-16: PR11 is not relevant to the refactor and is not a gate on it | Nothing. Do not re-raise this as a blocker |
| A9 | The compiled gate runs only `DSV4_MAX_MODEL_LEN=1024` | verified | P2 | `scratchpad/gate_dsv4_compile.sh:26`. It missed both of today's long-context failures — the real one and the self-inflicted one | Still worth a large-context case: 1024 did not catch the recursion-limit removal that broke 8192 |
| A10 | `max-num-batched-tokens` is the memory driver, not eager vs compiled | verified | — | at 9216 batched, eager KV was 126,516 tokens and execution OOM'd; at 1024 batched, eager 486,321 and compiled 485,523 | Keep 1024 for any long-context serving config |
| A13 | **Eager costs ~3× on generation** — 6.9 vs 20.4 tok/s median, and ~2× on prefill | verified | **P1** | median `Avg ... throughput` lines from `scratchpad/mmlupro_server.log` (eager) and `eval_results/phase3_limit50/vllm_server.log` (compiled). Matches the 3.6× end-to-end ratio | Stop using eager once A14 lands. This is the efficiency case for the single-graph backbone |
| A14 | **Compiled PASSES at 9216** on `d4f8c153` — KV 308,292 tokens, 33.45× concurrency, request served | **verified** | — | `PASS 9216 18:52:02`, `scratchpad/ceiling_9216_server.log` | Switch every serving script off `--enforce-eager`. Expect ~3× on generation (A13) |
| A15 | Probe cost is not weight loading: startup is 8.8 min and diffuse, largest milestone gap 30 s | verified | — | gap analysis over `mmlupro_server.log`, 16:26:35 → 16:35:22 | `--load-format dummy` would save little; do not expect it to speed up probing |
| B1 | Task 1 — temporary `[SCALE_PROBE]` logging in the FP8 linear path | verified | P2 | `fp8.py:234-244`, `:987-996` | Run 1, then grep `[SCALE_PROBE]` |
| B2 | Task 3 — removing `sys.setrecursionlimit(50000)` was **wrong and is reverted** | **refuted, then restored** | — | Run 1 passed at `DSV4_MAX_MODEL_LEN=1024` and I called it verified. At 8192 the compiled path dies in Dynamo AOT capture: repeating `forward_model → aot_compile → package.add_code → inspect.getsourcelines`, ending in `IndexError`/`OSError: lineno is out of bounds` (`scratchpad/ceiling_8192_server.log` 15:25:40). gxd3 was right on PR #464 | Restored with a comment naming the failure. True depth still unmeasured — the 50000 is inherited, not derived |
| B3 | Task 4 — weight cache moved off `/dev/shm` onto disk in 3 scripts | verified | P2 | `quant_weight_cache.py:56-62` resolves primary before legacy; last RAM run hit 162 GB | Run 1 |
| B4 | Task 5 — dead `patch_deepseek_v4_mla_cls` deleted | verified | P2 | no caller; whole-repo grep matched only the def. Discharges gxd3's PR #459 ask | Run 1 |
| B5 | Task 6 — `deepseek_v4_stubs.py` and its test deleted | verified | P2 | only importer was `test_dsv4_model.py:21` | Run 1 |
| B6 | Run 1 — compiled gate covering B1–B5 | **verified** | — | `run1_tasks13456`, 2026-08-16 13:58–14:30: `tracebacks=0 sweeps=13 raises=0 compile_lines=72`; generations and token IDs byte-identical to `gate_cache_compile.sub`; needle exact (`7391-QUARTZ-4408`) | B1–B5 are all verified. The recursion limit was not load-bearing |
| B7 | Warm-cache measurement — was the cache read, or rewritten? | **still open** | P2 | 132 GB / 344 files in `dsv4_quant_cache`, nothing in `/dev/shm`. Startup 32 min vs the baseline's 33 — consistent with a warm read, not proof. `quant_weight_cache.py` logs no hit/miss | Add a hit/miss log line before anyone quotes a speedup |
| C1 | Our requantizer is **numerically equivalent** to the reference, so this is a simplification, not a fix | verified | P2 | `scratchpad/compare_scale_equivalence.py`, CPU: routing through the standard block path gives scales **bit-identical** to tpu-inference (0.000e+00); our requantizer differs only by float32 rounding (6.9e-08). Our `axis=-1` on `[out,in]` ≡ reference `axis=0` on `[in,out]` | Delete `deepseek_v4_requant.py`, its call at `model.py:791`, and the override at `fp8.py:917-929`; let `weight_block_size:[128,128]` drive the standard path. Needs its own gate; generations predicted identical |
| C2 | The 2.71% Frobenius error is **not our defect** — it is inherent to per-channel scales and the reference pays it too | verified | — | same script: all three routes land on 2.4885% against a float64 block-dequantized ground truth (synthetic `wo_a`-shaped tensor) | Answers gxd3's PR #462 comment with a measurement |
| C3 | Both codebases end up per-channel at runtime; the divergence is structural only | verified | — | reference `layers/common/quantization/fp8.py:146-176` dequantizes the block scales then requantizes with `block_size=requant_block_size`, defaulting to `None` = per-channel. Its `_o_proj` views `wo_a.weight_scale` the same way ours does (`deepseek_v4_attention.py:218`) | Corrects the earlier claim that the reference stays block-quantized |
| D1 | Native model refactor — gxd3's headline ask, written locally, never pushed as a PR | written | **P1** | PR #462 thread; 9 commits `ff1033a2..e8ecf947` absent from `origin/main` | Split into PRs and send |
| D2 | Generic weight cache — PR #468 closed unmerged by you, content only on this branch | open | P2 | closed 2026-08-15T03:01; `quant_weight_cache.py` absent from `origin/main` | Decide resubmit vs drop |
| D3 | Review commitment: submit the `tpu_compiler.py` cache hardening as its own PR | closed, nothing owed | — | `git diff origin/main HEAD -- compilation/tpu_compiler.py` is empty. The earlier "54+/26− still on the branch" was diffed against the stale local `main` (4b87cff5) instead of `origin/main` (46a6a94b) | Nothing to submit. Whether the hardening landed elsewhere or was dropped is unknown |
| E1 | Perf gap vs tpu-inference: 1.27× at batch 1, 3.34× at batch 128 | open | P3 | from memory of a 20260809 run (unverified); `eval_results/perf_ref.log`, `perf_offline.log`, `perf_offline_nosync.log` exist and would settle it | Re-derive from those logs before quoting again |
| E2 | DP attention — sharding change through runner and KV cache manager | deferred | P3 | tpu-inference hard-validates MLA + DP, so no DP-off reference control exists | Needs its own planning pass |
| E3 | No genuine ours-vs-reference comparison has ever run — every "identical" compares us to ourselves | open | P3 | consequence of E2 | Unblocked only by E2 |
| F1 | `"DeepseekV4ROCMAiterMLAAttention"` string at `deepseek_v4_attention.py:253` | open | P3 | gxd3 asked on PR #459 to drop the ROCm dependency; the module-level import and `getattr` fallback he commented on are already gone | Remove if the tuple's other names can be shown not to match |
| G1 | **We still monkey-patch vLLM**: `model.py:147` does `HCHeadOp.forward_native = mhc_hc_head_native` | verified, **fully scoped** | **P1** | vLLM's `HCHeadOp.forward_native` raises `NotImplementedError` (only `forward_cuda`/`forward_xpu` exist), so a reimplementation *is* needed — but the reference does it through vLLM's supported out-of-tree hook: `tpu_inference/layers/vllm/custom_ops/mhc.py`, 131 lines, four `@X.register_oot` subclasses each defining `forward_tpu`. We already use `register_oot` in `gdn_attention_op.py`, `mla_attention_op.py`, `vision_attention.py` — DSv4's MHC is the lone outlier | See the G1 plan below |
| G2 | No `Head` class in the reference — it inlines `hc_head_op(...)` then `norm(...)` at the tail of `Model.forward` | verified | P2 | reference `deepseek_v4.py:656-664`; ours has a `DeepseekV4Head` class at `model.py:418` | Inline it to match, removing the extra class |
| G3 | File layout does not match `kimi_k3`: DSv4 is one 30 KB `model.py`, kimi splits into `layers.py`/`moe.py`/`model.py`/`attention.py` | verified | P2 | `ls models/vllm/kimi_k3/` vs `models/vllm/deepseek_v4/` | Split: `DeepseekV4MLP` + head + `mhc_*` → `layers.py`; `DeepseekV4MoE` → `moe.py`; keep decoder/model/causal-LM in `model.py`. Pure moves, but needs a re-gate |
| G5 | **DSv4 allocates KV outside the profiler's envelope**, unlike the shared path kimi uses | verified (divergence); cause of the util ceiling **inferred** | **P1** | `tpu_runner.py:5169-5258` takes `num_blocks` from the profiler and allocates `get_kv_cache_shape(num_blocks, spec.storage_block_size, …)` with no check that it fits `kv_cache_tensor.size`. The shared path at `:5375-5405` reconciles against `PallasAttentionBackend.get_kv_cache_page_size_bytes`, commenting that `spec.real_page_size_bytes` is 2× off on fp8. Symptom: at util 0.92 the DSv4 KV init asked for 2.19 GB with 1.60 GB free | Log `num_blocks × actual page bytes` vs `kv_cache_tensor.size` to confirm. Then fold DSv4 into the shared tail — Pate's own TODO at `:5364` says the same |
| G4 | DSv4 attention lives in `layers/vllm/custom_ops/deepseek_v4/`, while kimi keeps its attention inside the model package | verified | P3 | `models/vllm/kimi_k3/attention.py` vs `layers/vllm/custom_ops/deepseek_v4/deepseek_v4_attention.py` | Larger move than G3; decide whether kimi's placement is the target |

## Where the MMLU question stands — closed

Settled by A6 on 2026-08-16: MMLU-Pro scores 0.7893 ± 0.0241 on this branch
against 0.7786 ± 0.0154 on 2026-08-12, statistically indistinguishable, with the
same per-subject difficulty ordering. The model has not regressed. The
`mmlu_llama` score of 0.238 is an artifact of that task's serving path, and the
constant `C` in A4 is its signature, not the model's.

D1 is unblocked on accuracy. Whether it is blocked on the compile question
depends on A8, which is now weaker than when it was written — see the refuted
table. What is certain is that this branch cannot currently serve compiled at
long context; what is not certain is why, or that `de8a5a1c` is responsible.

## The open question, stated precisely

**Can this branch serve compiled above 1024 tokens, and if not, why?**

Known: 1024 compiled passes (run 1, and every gate). 8192 compiled failed on
Dynamo recursion with the limit removed — a self-inflicted failure now fixed.
9216 compiled failed once on XLA memory, under settings never since reproduced.

The next three steps, in order, each answering one thing:

1. **8192 compiled, limit restored** (running). Distinguishes "the 8192 failure
   was only the recursion limit" from "there is also a memory wall below 9216."
2. **9216 compiled at the probe's exact settings, limit restored.** The honest
   re-test of row C. Reproduces the 103.09 GB error or retires it.
3. **If it reproduces:** dump XLA buffer assignment
   (`XLA_FLAGS=--xla_dump_to=…`) at 9216 and at the 08-12 branch tip, and diff
   the largest live buffers. That names the cause instead of inferring it.

Do not conclude anything about `de8a5a1c` before step 3.

## G1 plan — replace the MHC monkey-patch with `register_oot`

Copy the reference's structure; the math is already ours and stays unchanged.

1. Add `src/vllm_torchtpu/layers/vllm/custom_ops/mhc.py` with four subclasses,
   mirroring `tpu_inference/layers/vllm/custom_ops/mhc.py`:
   `@MHCPreOp.register_oot class VllmMHCPreOp(MHCPreOp)` with `forward_tpu`, and
   the same for `MHCPostOp`, `HCHeadOp`, `MHCFusedPostPreOp`.
2. Move the bodies of `mhc_pre_torch`, `mhc_post_torch` and
   `mhc_hc_head_native` (`model.py:57-146`) into those `forward_tpu` methods
   verbatim. No numerics change is intended.
3. Delete `HCHeadOp.forward_native = mhc_hc_head_native` (`model.py:147`) and
   the three free functions.
4. In `DeepseekV4DecoderLayer.__init__`, hold `self.mhc_pre = MHCPreOp()` and
   `self.mhc_post = MHCPostOp()` and call them, as the reference does at
   `deepseek_v4.py:368-370, 379, 399`.
5. Ensure the new module is imported where the platform registers custom ops —
   see `platforms/tpu_platform.py:366`, which explains that `register_oot` only
   takes effect once the module is imported.

Risk: dispatch changes from a patched `forward_native` to `forward_tpu`, so the
call path differs even though the arithmetic does not. Needs a compiled gate;
generations are expected identical, and a diff would mean the bodies did not
transfer cleanly.

## Gap analysis, 2026-08-16

### A. Ours vs tpu-inference's DSv4

| Area | tpu-inference | ours | Verdict |
|---|---|---|---|
| Model classes | 5: `MLP`, `MoE`, `DecoderLayer`, `Model`, `ForCausalLM` | 6 — adds `DeepseekV4Head` | **G2** |
| MHC ops | four `@X.register_oot` subclasses with `forward_tpu`, in `layers/vllm/custom_ops/mhc.py` (131 lines) | monkey-patches `HCHeadOp.forward_native`, plus three module-level `mhc_*` functions in `model.py` | **G1** |
| KV cache allocation | shared path | bespoke `_initialize_ds_v4_kv_cache` branch that skips the profiler reconciliation | **G5** |
| Attention weight requant | none | none — deleted in `83abd731` | closed |
| Custom ops | `deepseek_v4_{attention,compressor,indexer}.py` | same three, same names | aligned |
| Kernels | 8 entries | 5 | **not a gap** — `compress_and_store/`, `compressor_v1.py` and `proj_and_save_state.py` form a self-contained cluster nothing outside it imports, in the reference either |
| Quant config | `deepseek_v4_fp8.py`, dispatches linear vs MoE | same file, same shape | aligned |
| Backbone compile | whole backbone precompiled (`_precompile_backbone_helper`, JAX) | whole backbone via `support_torch_compile` | aligned in intent |

### B. Ours vs `kimi_k3` in this repo

| Aspect | `kimi_k3` | DSv4 | Verdict |
|---|---|---|---|
| Files in model package | 5: `attention.py`, `layers.py`, `model.py`, `moe.py`, `kimi_vit.py` | 1: `model.py` | **G3** |
| `model.py` | 637 lines, 4 classes | 795 lines, 6 classes | **G3** |
| Attention location | inside the model package | `layers/vllm/custom_ops/deepseek_v4/` | **G4** |
| KV cache path | shared, via the general `_unified_kv_layout` flag | bespoke `_is_ds_v4` branch — the only model-specific allocation branch in the runner | **G5** |
| `gpu-memory-utilization` | 0.92–0.95 house standard | ≤0.83 | consequence of G5 |

**Distance summary.** Against the reference the remaining gaps are three: how MHC
is registered (G1), one extra class (G2), and KV allocation (G5). Everything
else is aligned, and the apparent kernel gap is illusory. Against `kimi_k3` the
gaps are layout (G3, G4) plus the same G5. G5 is the only one that costs
capacity rather than tidiness.

## Refuted — do not re-raise these

Each of these was asserted during investigation and then disproved. They are
recorded so the same ground is not re-covered.

| Claim | Verdict | What disproved it |
|---|---|---|
| "The model has regressed / may be producing noise" | **false** | MMLU-Pro 0.7893 ± 0.0241 on this branch vs 0.7786 ± 0.0154 on 08-12 |
| "tpu-inference keeps DSv4 linear layers block-quantized; we deviate by going per-channel" | **false** | the reference dequantizes the 128×128 scales and requantizes per-channel by default (`layers/common/quantization/fp8.py:146-176`, `requant_block_size=None`). Both end per-channel |
| "The 2.71% Frobenius error is a defect our requantizer introduced" | **false** | all three routes measure 2.4885% on the same input; it is inherent to per-channel |
| "The recursion limit is not load-bearing; run 1 verified its removal" | **false** | run 1 only exercises 1024 tokens. At 8192 the compiled path dies in Dynamo AOT capture (`ceiling_8192_server.log` 15:25:40). Restored |
| "`de8a5a1c` (one graph for the backbone) is the cause of the 9216 compile OOM" | **unproven, weakened** | the reference also compiles the whole backbone as one graph (`compilation_manager.py:_precompile_backbone_helper`). Structure alone does not explain it |
| "The 9216 run that worked was tpu-inference" | **false** | `eval_results/phase3_limit50/vllm_server.log` shows `Loading plugin torchtpu` and `vllm_torchtpu`, zero `tpu_inference`. It was our own 08-12 branch |
| "tpu-inference is not comparable because it doesn't use torch.compile" | **wrong framing** | it is the design being emulated and serves this model on this hardware. Its graph boundaries are a legitimate model to copy, even though it is JAX-jitted |
| "`lsof /dev/vfio/*` reliably shows whether the TPU is held" | **false** | it reported zero holders while a server was demonstrably running. Use `pgrep -af entrypoints.openai.api_server \| grep -v shell-snapshots` |
| "The `tpu_compiler.py` hardening is still owed as a standalone PR" | **false** | `git diff origin/main HEAD` on that file is empty; the earlier delta came from diffing against a stale local `main` |

## Run log

**Provenance warning.** Every 2026-08-16 run was branch
`patemotter/dsv4-native-model` at commit `e8ecf947`, with an *uncommitted working
tree that changed between runs*. The SHA therefore does not identify what ran —
`sys.setrecursionlimit(50000)` was absent for some rows and present for others at
the same commit. Reference tree: `tpu-inference` at `f329f3480`.

Going forward: commit before gating, or the run log's "RL" column is the only
thing distinguishing two runs that claim the same SHA.

### Exact invocations

```bash
# Shared by every run below
export PYTHONPATH=/home/patemotter_google_com/workspace/vllm-torchtpu/src
export VLLM_TP_SIZE=8 VLLM_EP_SIZE=8
export VLLM_QUANT_WEIGHT_CACHE_DIR=/home/patemotter_google_com/workspace/dsv4_quant_cache
VENV=/home/patemotter_google_com/workspace/uv_venv_dsv4
MODEL=/home/patemotter_google_com/workspace/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1

# mmlu_llama --log_samples diagnostic (rows: "mmlu_llama samples")
setsid nohup scratchpad/run_dsv4_mmlu_samples.sh

# MMLU-Pro (rows: mmlupro A-F; edit the script between attempts)
setsid nohup scratchpad/run_dsv4_mmlupro.sh
#   lm_eval --model local-chat-completions --tasks mmlu_pro --limit 20 \
#     --log_samples --apply_chat_template \
#     --gen_kwargs '{"chat_template_kwargs": {"enable_thinking": false}}' \
#     --model_args "...,num_concurrent=8,timeout=1800,max_retries=5"

# Compiled gate (row: run 1)
setsid nohup scratchpad/gate_dsv4_compile.sh . run1_tasks13456
#   sets DSV4_MAX_MODEL_LEN=1024 DSV4_MAX_NUM_BATCHED_TOKENS=1024
#        DSV4_MAX_NUM_SEQS=16 GPU_MEM_UTIL=0.48 VLLM_ENFORCE_EAGER=0

# Compile-ceiling probe (rows: ceiling 8192, 8192b)
setsid nohup scratchpad/probe_compile_ceiling.sh 8192

# Scale-equivalence comparison (CPU only, no chips)
JAX_PLATFORMS=cpu ${VENV}/bin/python scratchpad/compare_scale_equivalence.py
```

Outcomes below. "RL" is whether `sys.setrecursionlimit(50000)` was present.

| # | ctx | util | seqs | batched | mode | RL | outcome |
|---|---|---|---|---|---|---|---|
| 08-12 baseline | 9216 | ? | ? | 1024 | compiled | yes | MMLU-Pro 0.7786; KV 412,832 |
| mmlu_llama samples | 4096 | 0.48 | 16 | 4096 | eager | no | completed; exposed the constant-`C` artifact |
| mmlupro A | 9216 | 0.48 | 16 | 9216 | eager | no | startup `ValueError`: KV needs 47.8 GB, 26.29 available |
| mmlupro B | 9216 | 0.80 | 16 | 9216 | eager | no | KV 126,516; **execution OOM** |
| mmlupro C | 9216 | 0.75 | 8 | 1024 | **compiled** | **no** | **compile OOM, 103.09 GB HLO temporaries vs 94.74 GB** |
| mmlupro D | 9216 | 0.75 | 8 | 1024 | eager | no | KV 486,321; execution OOM |
| mmlupro E | 9216 | 0.55 | 8 | 1024 | eager | no | reached 265/280, died on 300 s client timeouts |
| mmlupro F | 9216 | 0.55 | 8 | 1024 | eager | no | **PASS** — 0.7893 ± 0.0241, `timeout=1800` |
| run 1 gate | 1024 | 0.48 | 16 | 1024 | compiled | no | **PASS**, generations byte-identical to baseline |
| ceiling 8192 | 8192 | 0.55 | 8 | 1024 | compiled | **no** | START_FAIL — Dynamo recursion, not memory |
| ceiling 8192b | 8192 | 0.55 | 8 | 1024 | compiled | **yes** | **stopped at 15:37 before a verdict** (Pate: ceiling work paused). No data |
| gate_09f6c694 | 1024 | 0.48 | 16 | 1024 | compiled | yes | **environmental failure, not code**: `PjRtClient is not initialized`. Eight orphaned `VLLM::Worker_*` from the killed ceiling probe still held the chips |
| gate_09f6c694b | 1024 | 0.48 | 16 | 1024 | compiled | yes | same, after clearing `psm_*` only — the orphans were the cause, not the segments |
| gate_09f6c694c | 1024 | 0.48 | 16 | 1024 | compiled | yes | **`tracebacks=0 sweeps=13 raises=0 compile_lines=72`, needle exact.** Generations **differ** from `gate_cache_compile.sub` — see below |
| ceiling 9216 | 9216 | 0.55 | 8 | 1024 | **compiled** | yes | **PASS** — KV 308,292, 33.45×, request served. Refutes A8 |
| mmlupro @ 09f6c694 | 9216 | 0.55 | 8 | 1024 | eager | yes | **0.7821 ± 0.0243** over 280. Matches 0.7893 (morning) and 0.7786 (08-12) within a fraction of a stderr. 114 min; one item hung ~35 min on a retry under `timeout=1800` |

| bench 0.92 | 10240 | 0.92 | 512 | 8192 | compiled | yes | **START_FAIL** — DSv4 KV init wanted 2.19 G, 1.60 G free |
| bench 0.85 | 10240 | 0.85 | 512 | 8192 | compiled | yes | **START_FAIL** — wanted 1009.6 M, 986.5 M free (short 23 M). KV 136,847 |
| bench 0.83 | 10240 | 0.83 | 512 | 8192 | compiled | yes | **START_FAIL** — wanted 978.4 M, **420.1 M** free (short 558 M). KV 132,611 |

### The utilization shortfall is non-monotonic

Lowering `gpu-memory-utilization` from 0.85 to 0.83 left *less* memory free at
the point of failure — 986 MB down to 420 MB — because a smaller budget shrinks
`num_blocks`, which shrinks the allocation, which chases the shrinking headroom.
There is no threshold to find by nudging, which is why DSv4 works at 0.55 and
fails at 0.83. Bisecting is the wrong instrument here; measure the accounting
disagreement instead (G5 step 1).

**No DSv4 throughput number was ever obtained.** The benchmark config exists at
`scripts/vllm/benchmarking/configs/deepseek-v4-flash-tp8-ep.sh` and has never
completed a run. The only throughput figures we have are the medians in A13,
harvested from serving logs.

### Why `09f6c694` moved generations

Commit `83abd731` was predicted to be byte-identical and was not. The prediction
was wrong for a reason the equivalence script could not see: the **old** path
quantized twice. `deepseek_v4_requant` converted block scales to per-channel over
the weight iterator, and then `_process_fp8_linear_weights` ran again on that
result — `block_quant` was False, so the per-channel branch dequantized to
float32 and requantized to fp8 a second time. The new path does one conversion.

So the change removed a redundant lossy round-trip. The new weights should be
*more* accurate, not merely different, but that is an argument and the drift is
real. The generations remain coherent and the needle control is exact.

Settled: MMLU-Pro on `09f6c694` scores **0.7821 ± 0.0243**, against 0.7893 the
same morning and 0.7786 on 08-12. The drift costs nothing in accuracy. Do not
treat byte-identity against `gate_cache_compile.sub` as the criterion for this
commit or later ones — that baseline predates the round-trip removal, and
`gate_09f6c694c.sub` supersedes it.

## Eager costs 3× — measured

Median throughput, from the servers' own `Avg ... throughput` lines:

| | prompt | generation | generation max |
|---|---|---|---|
| eager, `09f6c694`, 9216 ctx, 8 concurrent | 145.8 tok/s | **6.9 tok/s** | 9.6 |
| compiled, 08-12 baseline, 9216 ctx | 306.1 tok/s | **20.4 tok/s** | 53.5 |

**~3× on generation, ~2× on prefill.** The gap is larger on decode because
per-layer Python dispatch is paid 61 times per token with no large matmul to
hide behind; prefill does enough work per layer to dilute it. This is the
efficiency argument for compiling the backbone as one graph, and it matches the
3.6× end-to-end ratio (6.6 s/sample compiled vs 23.7 s/sample eager).

Every MMLU-Pro run so far has paid that, justified only by the one unreproduced
compile failure in run-log row C — which happened with the recursion limit
removed, at util 0.75 and seqs 16. As of 2026-08-16 18:36 the compiled probe at
9216 on the corrected tree has passed KV sizing (308,292 tokens, 33.45×
concurrency), which is well past where row C died. Provisional, pending verdict.

## Measuring memory properly

Running whole servers for a pass/fail bit is the wrong instrument. Better:

- **`XLA_FLAGS=--xla_dump_to=<dir>`** emits buffer-assignment output on
  *successful* compiles too, not only failures. That turns each probe into a
  measurement — peak bytes and the largest live buffers at a given context —
  instead of one bit. Already used in `tests/compilation/test_tpu_compiler.py:99`,
  so the flag works in this stack; the file contents are unread so far.
- **Fit, don't bisect.** With peak memory at three context sizes, fit
  `peak(ctx) = c + m·ctx` and solve for the 94.74 GB crossing. The slope-vs-
  intercept split is itself diagnostic: large `m` means activations scaling with
  context, large `c` means something fixed grew.
- **Harvest what is already printed.** Every startup logs `Available KV cache
  memory`, `GPU KV cache size` and `Maximum concurrency` — a free second
  measurement of the same budget.

Not a lever, though it looked like one: `--load-format dummy` would save little.
Startup is 8.8 min and **diffuse** — the largest gap between logged milestones is
30 s, spread across eight worker spawns, TPU init, weight read, profiling and KV
sizing. With a warm cache, requantization is not in the path at all.

Two confounds this log makes visible, both mine:

Row C — the only observation of the compile OOM — ran **without** the recursion
limit, at util 0.75 and seqs 16 differing from the probe's 0.55 and 8. It has
never been reproduced under the probe's settings with the limit restored. Until
it is, "compiled fails at 9216" rests on a single run with two uncontrolled
variables.

Rows B and D show `max-num-batched-tokens` dominates memory: at 9216 batched the
KV cache collapsed to 126,516 tokens, at 1024 batched it was 486,321. Any future
memory comparison must hold it fixed.

## Standing caveats

Merged PRs were reviewed on code structure. No review thread mentions accuracy,
and the 26/26 PASS in the August 14 handoff measures tracebacks, NaNs and decode
steps completed — not whether answers are correct. This model already shipped
all-zero MoE experts that produced fluent output for weeks, so fluent output is
not evidence.

Figures marked unverified come from memory or prior-session docs and were not
re-derived. Per the working agreement they are claims, not evidence.
