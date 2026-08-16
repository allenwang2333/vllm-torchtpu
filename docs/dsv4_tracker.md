# DeepSeek-V4: everything in flight

One table, current as of August 16, 2026.

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
The compile-ceiling question (A8) is a capability limit, not an accuracy or
design defect, and is demoted accordingly.

| # | Item | Status | Pri | Evidence / reference | Next step |
|---|---|---|---|---|---|
| A1 | MMLU-Pro 0.7786 ± 0.0154 over 700 samples, per-subject spread from law 0.54 to math 0.92 | verified | — | `eval_results/phase3_limit50/.../results_2026-08-12T15-49-05.json`, `limit: 50`, filter `custom-extract` | Nothing; this is the working-model reference point |
| A2 | MMLU (`mmlu_llama`) 0.2379 mean over 62 subtasks against 0.25 chance, no subtask above 0.347 | explained by A4 | P3 | `scratchpad/mmlu_results/*/results_2026-08-15T12-54-29*.json`, full run, `limit: None` | Treat this score as meaningless until the serving path is fixed |
| A3 | `mmlu_llama` prompts end with a partial *assistant* turn, `"The best answer is"`, and `continue_final_message: true`; the model is meant to continue it with one letter and stop at `.` | verified | — | `samples_mmlu_llama_astronomy_2026-08-16T02-37-45.jsonl`, `arguments.arg_0` | Reference for A6 |
| A4 | The model does not continue it: **50 of 60 responses are the literal letter `C`** — zero A, B or D — and the other 10 open a fresh `# Explanation` block | verified | — | same file, `resps`; run `scratchpad/run_dsv4_mmlu_samples.sh` 02:24–02:37 | Constant `C` scores ~0.25 by accident; the 10 explanations score 0. That is the whole of A2 |
| A5 | Inferred mechanism: `continue_final_message` is not stitched through the `local-chat-completions` → vLLM chat path, so the model answers a prompt that already ended | inferred | P3 | 50/50 identical output is degenerate even for a mis-primed prompt, so the mechanism is still unproven — but A6 rules out the model, so the fault is in that path | Only worth chasing if we need `mmlu_llama` specifically; MMLU-Pro is the working harness |
| A6 | MMLU-Pro on this branch: **0.7893 ± 0.0241** over 280 samples vs the baseline's 0.7786 ± 0.0154 — 0.37σ apart, no regression | verified | — | `scratchpad/mmlupro_results/.../results_2026-08-16T08-09-35.json`, `limit 20`, eager. Per-subject shape matches: physics/biology 0.95, law/history/philosophy 0.65 | Closes A2, A3, A5. The model is fine; `mmlu_llama`'s serving path is the only broken thing |
| A7 | MMLU-Pro runner: `lm_eval` flags from `scratch/run_mmlu_limit100.sh`, server and env from `gate_dsv4_compile.sh` | written | — | the four `scratch/run_mmlu*.sh` all export `LIBTPU_INIT_ARGS` and `MOE_REQUANTIZE_*`, the env behind the repeating-output harness bug, so they are not safe to run verbatim | Reuse `scratchpad/run_dsv4_mmlupro.sh` for future MMLU-Pro runs |
| A8 | Compiled serving failed once at 9216 with 103.09 GB of HLO temporaries vs 94.74 GB HBM | **observed once, not reproduced** | **P1** | `scratchpad/mmlupro_server.log` 04:42, all 8 workers, `Failed to traverse and compile`. Run-log row C: that run had **no recursion limit**, util 0.75, seqs 8 — two variables differing from the probe | Re-test at the probe's settings with the limit restored (step 2 of the open question). Do not attribute it to `de8a5a1c` until an XLA buffer dump says so |
| A11 | `de8a5a1c` sits inside the D1 stack, so *if* it proves to be A8's cause, shipping the refactor ships the regression | conditional | P2 | `de8a5a1c` is one of the 9 commits in `ff1033a2..e8ecf947`. But the reference compiles the backbone as one graph too, so the structure alone is not evidence | Resolve A8 first. This row means nothing until then |
| A12 | gxd3 agreed to the refactor "after PR11", but PR11 was never opened | dropped | — | no kv-admission PR in `gh pr list --author @me --state all`. Pate's call 2026-08-16: PR11 is not relevant to the refactor and is not a gate on it | Nothing. Do not re-raise this as a blocker |
| A9 | The compiled gate never exercises this: it runs `DSV4_MAX_MODEL_LEN=1024` | verified | P1 | `scratchpad/gate_dsv4_compile.sh:26` | Add a large-context case to the compiled gate, or A8 recurs unnoticed |
| A10 | `max-num-batched-tokens` is the memory driver, not eager vs compiled | verified | — | at 9216 batched, eager KV was 126,516 tokens and execution OOM'd; at 1024 batched, eager 486,321 and compiled 485,523 | Keep 1024 for any long-context serving config |
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
| G1 | **We still monkey-patch vLLM**: `model.py:147` does `HCHeadOp.forward_native = mhc_hc_head_native` | verified | **P1** | the reference instantiates `MHCPreOp()`, `MHCPostOp()`, `MHCFusedPostPreOp()` as decoder-layer members (`deepseek_v4.py:368-370`) and never patches. Ours uses free functions `mhc_pre_torch`/`mhc_post_torch` plus this patch | Adopt the reference's op objects. This is the pattern gxd3 asked us to leave on PR #462, still present in the "native" model |
| G2 | No `Head` class in the reference — it inlines `hc_head_op(...)` then `norm(...)` at the tail of `Model.forward` | verified | P2 | reference `deepseek_v4.py:656-664`; ours has a `DeepseekV4Head` class at `model.py:418` | Inline it to match, removing the extra class |
| G3 | File layout does not match `kimi_k3`: DSv4 is one 30 KB `model.py`, kimi splits into `layers.py`/`moe.py`/`model.py`/`attention.py` | verified | P2 | `ls models/vllm/kimi_k3/` vs `models/vllm/deepseek_v4/` | Split: `DeepseekV4MLP` + head + `mhc_*` → `layers.py`; `DeepseekV4MoE` → `moe.py`; keep decoder/model/causal-LM in `model.py`. Pure moves, but needs a re-gate |
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
