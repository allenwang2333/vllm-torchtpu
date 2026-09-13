# Qwen3.8-Flash-Next (`qwen4_exp`) Out-of-Tree TPU Port — Status & Repro Guide

## 1. Summary & Current Repository State

This branch (`iharsh/qwen3.8-flash-next-support`) refactors `Qwen/Qwen3.8-Flash-Next` (`qwen4_exp`) TPU support to a **strictly out-of-tree architecture** with zero model-specific directories under `src/vllm_torchtpu/models/vllm/`.

### Modified / Added Files in Repository
1. **[`src/vllm_torchtpu/layers/vllm/custom_ops/qwen4_exp_ops.py`](file:///usr/local/google/home/iharsh/all-repos/vllm-torchtpu/src/vllm_torchtpu/layers/vllm/custom_ops/qwen4_exp_ops.py)**
   - Single out-of-tree adapter that patches upstream `vllm.models.qwen4_exp.nvidia.*` at runtime.
   - Implements PyTorch/XLA replacements for CUDA/Triton custom ops (`grouped_gemma_rmsnorm`, `hc_silu`, `hc_gate_mix`, `hc_combine`, `hc_combine_norm`, `ple_gate`, `ple_conv`, `ple_ngram_ids`).
   - Adapts `Qwen4ExpPLELayer` and `Qwen4ExpModel` to run without Mamba KV-cache allocation and supports `--hf-overrides '{"architectures": ["Qwen4ExpForCausalLM"]}'` (ignoring vision tower weights when serving causal LM directly).
2. **[`src/vllm_torchtpu/platforms/tpu_platform.py`](file:///usr/local/google/home/iharsh/all-repos/vllm-torchtpu/src/vllm_torchtpu/platforms/tpu_platform.py)**
   - Calls `register_qwen4_exp_config()` and registers `Qwen4ExpForCausalLM` and `Qwen4ExpForConditionalGeneration` in `ModelRegistry`.
3. **[`tests/models/vllm/test_qwen4_exp_model.py`](file:///usr/local/google/home/iharsh/all-repos/vllm-torchtpu/tests/models/vllm/test_qwen4_exp_model.py)**
   - Out-of-tree unit test suite verifying custom ops, attention, and model forward pass (`4 passed`).
4. **[`.buildkite/vllm_lkg.version`](file:///usr/local/google/home/iharsh/all-repos/vllm-torchtpu/.buildkite/vllm_lkg.version)**
   - Pinned to upstream `vllm` commit `3116c5d06bfe76501b3dd6b5434bfc7f3274f5e7`.

---

## 2. What Works vs. What Fails

| Verification Target | Status | Details |
| :--- | :--- | :--- |
| **Local Unit Tests (`pytest`)** | **PASS** | `pytest tests/models/vllm/test_qwen4_exp_model.py` passes (`4 passed in 4.95s`). |
| **Offline TP=8 Weight Loading Parity** | **PASS** | All parameters and buffers across all 8 TP ranks match the working in-tree reference (`/tmp/in_tree_pkg/` on `jax-tpu-pod`) with `max diff = 0.0`. |
| **Offline TP=8 Single-Step Forward Parity** | **PASS** | Single-step forward pass logits with `ple_layer_ids=[1, 2]` match the working in-tree reference with `max diff = 0.0` across all 8 TP ranks. |
| **`--hf-overrides '{"architectures": ["Qwen4ExpForCausalLM"]}'`** | **PASS** | Server starts cleanly in text-only mode without loading vision tower weights. |
| **Live TP=8 Prefill (Token 0)** | **PASS** | First generated token (Token 0) is accurate (`" Paris"` for `"The capital of France is"`, `"The"` for `"What is the capital of France?"`). |
| **Live TP=8 Multi-Token Decode (Token 1+)** | **FAIL** | Starting at Token 1 (first decode step), output tokens become corrupted (`" Paris效 is_biosa gn worrying..."`). |

---

## 3. Root Cause Analysis & Things Tried

### Why Token 0 (Prefill) Works and Token 1+ (Decode) Corrupts
1. **In-Tree Reference vs. Out-of-Tree Adapter**:
   - The earlier in-tree implementation (`/tmp/in_tree_pkg/` on `jax-tpu-pod`) produced `"The capital of France is **Paris**."` on `/v1/chat/completions` and `89.12%` on 5-shot MMLU.
2. **Offline TP=8 Parity Tests (`tp8_param_diff.py` & `test_tp8_forward_parity_ple.py`)**:
   - Ruled out parameter sharding, weight loading, RoPE, `Qwen4ExpQSAAttention`, `Qwen4ExpSparseMoeBlock`, and HyperConnection math on a single forward pass (`max diff = 0.0`).
3. **Why Multi-Step Decode Diverges on TPU**:
   - In `vllm_torchtpu`, `tpu_runner.py` precompiles backbone token buckets during `_precompile_backbone` (bucket sizes `16`, `32`, `64`, ...) using `_dummy_run(num_tokens=16, ...)` where `seq_lens = [16]` and `length = 16` (`is_new_prefill = True`).
   - When `_prepare_tpu_ple_inputs` and `_tpu_short_conv` in `qwen4_exp_ops.py` branch on Python booleans (`if is_new_prefill:`), XLA tracing during `_dummy_run` bakes `is_new_prefill = True` (i.e. `.fill_(eos_id)` on `_ngram_context_cache` and `.zero_()` on `_conv_history_cache`) into the compiled graph for the `num_tokens=16` bucket.
   - On decode steps (`1` token padded to the `num_tokens=16` bucket), the traced `num_tokens=16` graph replays the baked-in cache reset/padding logic instead of preserving the request's n-gram and short-conv history across decode steps.
   - In addition, `AttentionMetadata` fields (`query_start_loc`, `seq_lens`, `mamba_state_indices`) on TPU are `jax.Array` instances (not `torch.Tensor`) and include padded request entries (`query_start_loc_np[num_reqs+1:] = query_start_loc_np[num_reqs]`).

---

## 4. Exact Reproduction Commands

### 4.1 Local Unit Tests (Cloudtop)
```bash
pytest tests/models/vllm/test_qwen4_exp_model.py
```

### 4.2 Sync Patch from Cloudtop to Remote TPU VM (`jax-tpu-pod`)
```bash
git diff HEAD > /tmp/oot_qwen4.patch
scp /tmp/oot_qwen4.patch jax-tpu-pod:/tmp/oot_qwen4.patch
ssh jax-tpu-pod "cd /mnt/v7x8-disk/vllm-torchtpu && git checkout -- . && git clean -fd && git apply /tmp/oot_qwen4.patch"
```

### 4.3 Start TP=8 Server on `jax-tpu-pod` (Text Decoder Only)
```bash
ssh jax-tpu-pod "pkill -9 -f 'vllm serve' || true; pkill -9 -f 'VLLM::EngineCore' || true"
ssh jax-tpu-pod "cd /mnt/v7x8-disk/vllm-torchtpu && TMPDIR=/mnt/v7x8-disk/tmp HF_HOME=/mnt/v7x8-disk/vllm-cache /usr/local/bin/vllm serve Qwen/Qwen3.8-Flash-Next \
  --tensor-parallel-size 8 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.65 \
  --port 8000 \
  --hf-overrides '{\"architectures\": [\"Qwen4ExpForCausalLM\"]}' > /tmp/vllm_oot_causal_serve.log 2>&1 &"
```

### 4.4 Test Direct Completion & Chat Completion
```bash
# Test 1: Direct completion (Token 0 is " Paris", Token 1+ corrupts)
ssh jax-tpu-pod "curl -s http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    \"model\": \"Qwen/Qwen3.8-Flash-Next\",
    \"prompt\": \"The capital of France is\",
    \"max_tokens\": 16,
    \"temperature\": 0.0
  }'"

# Test 2: Chat completion with thinking disabled
ssh jax-tpu-pod "curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    \"model\": \"Qwen/Qwen3.8-Flash-Next\",
    \"messages\": [{\"role\": \"user\", \"content\": \"What is the capital of France?\"}],
    \"max_tokens\": 32,
    \"temperature\": 0.0,
    \"chat_template_kwargs\": {\"enable_thinking\": false}
  }'"
```

### 4.5 Run MMLU Benchmark (`lm_eval`)
```bash
ssh jax-tpu-pod "TMPDIR=/mnt/v7x8-disk/tmp HF_HOME=/mnt/v7x8-disk/vllm-cache lm_eval \
  --model local-completions \
  --model_args model=Qwen/Qwen3.8-Flash-Next,base_url=http://localhost:8000/v1/completions,num_concurrent=16,max_retries=3,tokenized_requests=False \
  --tasks mmlu_llama \
  --num_fewshot 5 \
  --batch_size auto"
```

### 4.6 Recommended Next Fix for Decode Parity
In [`src/vllm_torchtpu/layers/vllm/custom_ops/qwen4_exp_ops.py`](file:///usr/local/google/home/iharsh/all-repos/vllm-torchtpu/src/vllm_torchtpu/layers/vllm/custom_ops/qwen4_exp_ops.py):
- Avoid Python branching (`if is_new_prefill:`) and `.cpu().tolist()` inside `_prepare_tpu_ple_inputs` and `_tpu_short_conv`, or mark `_prepare_tpu_ple_inputs` / `_tpu_short_conv` so XLA does not bake the `_dummy_run` prefill branch (`seq_len <= length`) into the `num_tokens=16` bucket executable used during decode steps.
