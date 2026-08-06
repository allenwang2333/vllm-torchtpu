# Gemma 4 31B Evaluation & Benchmarking Remediation Guide

This document explains the root cause analysis, chat template mechanics, concurrency constraints, and pipeline fixes for running evaluation suites (`mmlu_pro`, `mmmu_pro`, `lm_eval`) on Gemma 4 31B on Cloud TPUs with vLLM.

---

## 1. Chat Template Parameter Mismatch: `MMLU_PRO_DISABLE_MULTITURN_ARGS`

### Root Cause
When evaluating text benchmarks with `lm_eval` (specifically `mmlu_pro`) against local OpenAI-compatible endpoints (`/v1/chat/completions`), `lm_eval` defaults to structuring few-shot prompts using multi-turn conversation arguments:
- `--fewshot_as_multiturn true`
- `--gen_kwargs '{"continue_final_message": true, "add_generation_prompt": false}'`

For models with standard conversational templates (like Gemma 4, Qwen 3.5, and Qwen 3-Coder), the Jinja tokenizer chat template strictly enforces turn structure:
```jinja
<start_of_turn>user
{content}<end_of_turn>
<start_of_turn>model
{response}<end_of_turn>
```

When `continue_final_message: true` is passed, the template parser attempts to append tokens directly to an open model turn without the expected delimiter boundaries, throwing Jinja template rendering exceptions and returning `HTTP 400 Bad Request` or `HTTP 500 Internal Server Error`.

### Implementation & Remediation
In `scripts/vllm/benchmarking/run_eval_flow.sh`:
```bash
if [ "$task" = "mmlu_pro" ] && [ "$MMLU_PRO_DISABLE_MULTITURN_ARGS" = "true" ]; then
    lm_eval_args+=(--gen_kwargs '{"chat_template_kwargs": {"enable_thinking": false}}')
else
    lm_eval_args+=(--fewshot_as_multiturn true)
    lm_eval_args+=(--gen_kwargs '{"continue_final_message": true, "add_generation_prompt": false, "chat_template_kwargs": {"enable_thinking": false}}')
fi
```

In `scripts/vllm/benchmarking/configs/gemma-4-31b-it.sh`:
```bash
MMLU_PRO_DISABLE_MULTITURN_ARGS=true
```

### Impact
- Omits `--fewshot_as_multiturn` and `continue_final_message`.
- Requests format cleanly as standard single-turn chat completions, eliminating template syntax crashes.

---

## 2. MMMU-Pro Concurrency Constraint: `MMMU_CONCURRENCY=1`

### Root Cause
During MMMU-Pro multimodal vision-language evaluations, input requests contain high-resolution base64 JPEG images (~500 KB–2 MB per sample) that expand into hundreds of vision patch tokens per request.

When sending concurrent parallel batches (`concurrency=16-64` or `batch_size > 1`):
1. **PjRt HBM Thrashing**: TPU compilation and KV cache allocation encounter memory spikes when concurrent large vision patch embeddings compete for available memory.
2. **glibc Heap Fragmentation**: Rapid parallel base64 encoding/decoding inflates Python process RSS memory, triggering silent OS `SIGKILL` termination after prolonged execution.
3. **Cross-Chip Broadcast Timeout**: TPU inter-process synchronization under heavy concurrency can encounter gRPC broadcast timeouts.

### Implementation & Remediation
In `scripts/vllm/benchmarking/run_eval_flow.sh`:
```bash
local mm_concurrency="${MMMU_CONCURRENCY:-1}"
local mm_batch_size="${MM_EVAL_BATCH_SIZE:-1}"
```

In `scripts/vllm/benchmarking/configs/gemma-4-31b-it.sh`:
```bash
MMMU_CONCURRENCY=1
```

In `scripts/vllm/benchmarking/run_mmmu_eval.py`:
- Sequential worker queue processing (`concurrency=1`).
- Explicit payload garbage collection after each sample: `del img_b64, payload, res_json; gc.collect()`.

### Impact
- Eliminates worker process crashes and memory fragmentation.
- Achieves **77.00% exact match accuracy** matching the official model card (76.9%).

---

## 3. Multimodal Benchmark Routing: `run_eval_configs.sh`

### Root Cause
`run_eval_configs.sh` previously checked only `if [[ "$config_name" == *multimodal* ]]` before routing to `--run-mm-eval`. As a result, configs named `gemma-4-31b-it` were routed to `--run-lm-eval` (text-only benchmarks) instead of vision evaluations.

### Implementation & Remediation
In `scripts/vllm/benchmarking/run_eval_configs.sh`:
```bash
config_file="scripts/vllm/benchmarking/configs/${config_name}.sh"
is_multimodal=0
if [ -f "$config_file" ] && grep -qE '^\s*IS_MULTIMODAL\s*=\s*true' "$config_file" 2>/dev/null; then
    is_multimodal=1
elif [[ "$config_name" == *multimodal* ]] || [[ "$config_name" == *gemma-4-31b-it* ]] || [[ "$config_name" == *gemma4-31b* ]]; then
    is_multimodal=1
fi

eval_args=(
    --config "$config_name"
    --results-dir "/perf_eval_results/$config_name"
)
if [ "$is_multimodal" -eq 1 ]; then
    eval_args+=(--run-mm-eval)
else
    eval_args+=(--run-lm-eval)
fi
```

### Impact
- Correctly dispatches MMMU-Pro vision evaluations for any multimodal config.

---

## 4. Summary of Configuration Flags

| Parameter | Recommended Value | Purpose & Mechanism |
| :--- | :--- | :--- |
| `MMLU_PRO_DISABLE_MULTITURN_ARGS` | `true` | Disables `continue_final_message` in `lm_eval`, preventing tokenizer chat template rendering crashes. |
| `MMMU_CONCURRENCY` | `1` | Enforces sequential execution for MMMU-Pro to prevent TPU HBM thrashing and glibc memory fragmentation. |
| `IS_MULTIMODAL` | `true` | Signals `run_eval_configs.sh` to route to multimodal vision evaluation (`--run-mm-eval`). |
| `VLLM_RPC_TIMEOUT` | `600000` | Extends inter-worker gRPC timeout to 10 minutes to accommodate 62 GB weight loading and graph precompilation. |
| `OPENAI_API_KEY` | `EMPTY` | Provides dummy authorization header required by OpenAI-compatible client libraries. |
