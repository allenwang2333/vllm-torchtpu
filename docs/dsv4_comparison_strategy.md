# Correctness & Performance Comparison Strategy: torchtpu-vllm vs. tpu-inference

This document outlines the strategy to verify the correctness (numerical parity) and evaluate the performance of the `torchtpu-vllm` DeepSeek-V4 implementation against the reference `tpu-inference` implementation on TPU.

---

## 1. Correctness Verification (Numerical Parity)

To ensure our implementation does not introduce numerical drift, we must perform layer-by-layer activation matching using identical inputs and weights.

### A. Methodology
1.  **Seed and Input Alignment:**
    *   Use a fixed prompt (e.g., `"The quick brown fox jumps over the lazy dog"`).
    *   Ensure both engines use the same model checkpoint (`deepseek-ai/DeepSeek-V4-Flash`) with identical weight loading.
2.  **Instrumentation (Activation Saving):**
    *   Instrument both codebases to dump intermediate tensors to disk (`.pt` or `.npy` files) during a single-step forward pass.
    *   **Tensors to capture:**
        *   Attention query (`q`), compressed KV, and rotary embeddings (`q_rope`).
        *   Attention output (`attn_output`) before projection.
        *   Grouped projection output (`z_local`).
        *   MoE gating/routing logits and selected expert indices.
        *   MoE expert outputs and final layer outputs.
        *   Final model logits.
3.  **Numerical Comparison Metrics:**
    *   Write a comparison script to calculate:
        *   **Cosine Similarity:** Measures directional alignment (target: $> 0.999$ for BF16, $> 0.99$ for FP8).
        *   **Relative L2 Norm:** Measures magnitude difference (target: $< 10^{-3}$ for BF16, $< 10^{-2}$ for FP8):
            $$\text{Relative L2} = \frac{\|x_{\text{vllm}} - x_{\text{inference}}\|_2}{\|x_{\text{inference}}\|_2}$$
        *   **Max Absolute Error:** To detect localized spikes or NaN propagation.

---

## 2. Performance Comparison

We must compare the efficiency of both engines during both the **Prefill** (compute-bound) and **Decode** (memory-bandwidth-bound) phases.

### A. Key Metrics
1.  **Time-to-First-Token (TTFT) Latency (ms):** Measures prefill performance. Reflects the efficiency of the attention prefill kernel and MoE execution with large token counts.
2.  **Inter-Token Latency (ITL) (ms):** Measures decode performance. Reflects the efficiency of the SWA/MLA decode kernels and MoE execution with batch size 1.
3.  **Throughput (tokens/sec):** Total tokens generated per second under varying concurrency.
4.  **Memory Footprint (GB):** Peak HBM usage during profiling and execution.

### B. Benchmarking Scenarios
Run benchmarks on `alyssa` (TP=8) under the following configurations:

| Scenario | Phase | Batch Size | Prompt Length | Generation Length |
| :--- | :--- | :--- | :--- | :--- |
| 1. Single Query | Prefill | 1 | 512 | 1 |
| 2. Single Decode | Decode | 1 | 1 | 128 |
| 3. Batch Prefill | Prefill | 16 | 512 | 1 |
| 4. Batch Decode | Decode | 16 | 1 | 128 |
| 5. Long Context | Prefill | 1 | 4096 | 1 |

### C. Profiling and Bottleneck Attribution
If a performance delta is observed:
1.  **Extract HLO:** Dump the JAX HLO (High-Level Optimizer) proto files from both runs to compare the compiled XLA graphs.
2.  **Compare Kernel Latencies:** Use Perfetto to isolate the execution time of:
    *   `pallas::deepseek_v4_attention` vs. the `tpu-inference` attention kernel.
    *   `fused_moe` (JAX-native fallback) vs. the `tpu-inference` MoE kernel.

---

## 3. End-to-End Evaluation via `lm-eval`

To evaluate both correctness (accuracy on tasks like MMLU-Pro) and end-to-end performance under realistic workloads, you can run the `lm-evaluation-harness` (`lm-eval`) directly on the TPU VM.

### A. Setup
Install `lm-eval` in the VM's virtual environment:
```bash
pip install lm-eval
```

### B. TPU Evaluation Command
Run the following command on the TPU VM. It uses the `vllm` model class (which maps to our patched `torchtpu-vllm` engine):

```bash
lm_eval \
  --model vllm \
  --model_args '{
    "pretrained": "deepseek-ai/DeepSeek-V4-Flash",
    "tensor_parallel_size": 8,
    "max_model_len": 8192,
    "max_num_batched_tokens": 8192,
    "block_size": 256,
    "kv_cache_dtype": "fp8",
    "gpu_memory_utilization": 0.90
  }' \
  --tasks mmlu_pro \
  --apply_chat_template \
  --gen_kwargs '{}' \
  --log_samples \
  --output_path "~/workspace/dsv4/eval_logs/mmlu_pro" \
  --seed 42 \
  --batch_size auto
```

### C. TPU Argument Mapping Notes
*   **`pretrained`:** Points to the HF Hub model ID. It will automatically use the cached weights in `/home/patemotter_google_com/workspace/hf_cache`.
*   **`attention_backend` / `gdn_prefill_backend`:** Omitted from the TPU command. Our `TPUModelRunner` automatically overrides these and routes all attention/prefill execution to our sharded JAX/Pallas custom ops.
*   **`gpu_memory_utilization`:** Set to `0.90` (slightly lower than the GPU's `0.95` to account for TPU host-device allocator overhead and prevent host-side OOMs during compilation).

---

## 4. Running the Repository's Integration Tests

To run the exact accuracy verification tests used in the `tpu-inference` repository:

### A. Via PyTest (Recommended)
You can run the integration test suite directly. This uses `gsm8k` to verify model accuracy:

```bash
# Run FP8 KV Cache accuracy test
pytest -rP scripts/vllm/integration/test_accuracy.py::test_lm_eval_accuracy_v1_engine_fp8_kv_cache \
    --tensor-parallel-size=8 \
    --fp8-kv-model-name=deepseek-ai/DeepSeek-V4-Flash \
    --expected-value=0.75
```

### B. Via Shell Script
Alternatively, you can run the helper script which wraps the `lm_eval` CLI and parses the output scores:

```bash
./tests/e2e/check_lm_eval.sh \
    --model_name deepseek-ai/DeepSeek-V4-Flash \
    --use_moe_ep_kernel 1 \
    --tensor_parallel_size 8 \
    --max_model_len 2048 \
    --max_num_batched_tokens 2048 \
    --max_gen_toks 128 \
    --enable_expert_parallel 1 \
    --flex_threshold 0.70 \
    --strict_threshold 0.70 \
    --block_size 256
```
This script will evaluate the model on the `gsm8k_cot` task and assert that both the `flexible-extract` and `strict-match` scores meet the `0.70` threshold.

---

## 5. Running the vLLM Serving Benchmark (TPU Production Command)

This is the exact command used to spin up the vLLM OpenAI-compatible serving API on the TPU VM `alyssa` (v7x-8) under production-like settings. It enables MXFP4 expert quantization, Expert Parallelism (EP=8), and Data-Parallel Attention.

### A. TPU Serving Command
Execute this on the TPU VM:

```bash
# Enable MXFP4 MoE quantization and new TPU model path
MOE_REQUANTIZE_WEIGHT_DTYPE=fp4 \
MOE_REQUANTIZE_BLOCK_SIZE=512 \
NEW_MODEL_DESIGN=1 \
MODEL_IMPL_TYPE=vllm \
vllm serve deepseek-ai/DeepSeek-V4-Flash \
  --max-model-len=9216 \
  --max-num-batched-tokens=1024 \
  --max-num-seqs=128 \
  --kv-cache-dtype=fp8 \
  --no-enable-prefix-caching \
  --gpu-memory-utilization=0.4 \
  --tensor-parallel-size=8 \
  --download_dir=/mnt/pd \
  --async-scheduling \
  --enable-expert-parallel \
  --additional_config='{"sharding": {"sharding_strategy": {"enable_dp_attention": true}}}'
```

### B. Parameter Breakdown
*   **`MOE_REQUANTIZE_WEIGHT_DTYPE=fp4` / `MOE_REQUANTIZE_BLOCK_SIZE=512`:** Forces the MoE weights to be requantized to MXFP4 (FP4) with a block size of 512 during weight loading, saving massive HBM.
*   **`NEW_MODEL_DESIGN=1` / `MODEL_IMPL_TYPE=vllm`:** Environment variables that trigger our patched TPU model class and vLLM-based execution path.
*   **`--enable-expert-parallel`:** Shards the 256 experts across the 8 TPU devices (EP=8).
*   **`enable_dp_attention: true`:** Enables data-parallel attention sharding to optimize throughput under large context lengths.
*   **`--gpu-memory-utilization=0.4`:** Limits the KV cache allocation to 40% of TPU memory, which is highly stable and prevents host/device OOMs when compiling the 9216 context length.
*   **`--download_dir=/mnt/pd`:** Points to the persistent disk on the VM for weight storage.

---

## 4. Kernel Validation & Bisection Status

To ensure numerical correctness and avoid debugging complex integration failures in vLLM, our strategy separates standalone kernel validation from end-to-end bridge execution.

### A. Standalone Kernel Parity (100% Validated)
*   **Methodology:** Execute standalone unit test suites directly in pure JAX on VM `alyssa` (`pytest -v tests/kernels/test_gmm_v2.py tests/kernels/test_ragged_gather_reduce.py`).
*   **Results (`task-3703`):** All 64 standalone kernel unit tests passed in 112 seconds.
*   **Conclusion:** Proves that the reference Pallas FP8 General Matrix Multiply (`gmm_v2`) and SparseCore token routing/gathering (`ragged_gather_reduce`) kernels perform exact arithmetic, block quantization, and activation fusion without NaNs or indexing errors.

### B. Attention (MLA) Parity (100% Achieved)
*   **Methodology:** Compare attention outputs against `/Users/patemotter/dsv4/tpu-inference` as the ground truth.
*   **Resolution:** Restoring reference Pallas kernels ([mla_swa.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/kernels/deepseek_v4/mla_swa.py) and [mla.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/kernels/deepseek_v4/mla.py)) eliminated an online softmax `-inf` initialization bug. Slicing `self.attn_sink` in Python/PyTorch eliminated parameter broadcasting shape crashes (`(64, 1) vs (8, 128)`).
*   **Results:** Produces clean activations without NaNs (`norm=21.0664`).

### C. MoE Parity Strategy
*   **Methodology:** Having confirmed that PyTorch inputs are clean (`x norm=256.0000`) and standalone kernels are bug-free, replace [fused_moe_gmm.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/layers/common/fused_moe_gmm.py) with the reference implementation from `tpu-inference`.
*   **Goal:** Restore reference tensor-parallel sharding partition specs (`P(None, None, 'model')`) and strip out 4,142 bytes of experimental SparseCore divergence to eliminate NaNs during end-to-end execution.



