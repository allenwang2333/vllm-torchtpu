# DeepSeek-V4 Granular Re-implementation Plan

This document outlines the step-by-step plan to re-implement DeepSeek-V4 support on the clean branch `patemotter-dsv4-clean`, using small, verified commits.

---

## Final Clean Commit History

The entire DeepSeek-V4 integration branch has been squashed into 5 clean, logical, and concise commits:

### Commit 1: Add JAX Kernels and Unit Tests [STATUS: COMPLETED]
*   **Commit Message:** `feat(deepseek_v4): add JAX kernels and unit tests`
*   **Files Added:**
    *   [compress_norm_rope.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/kernels/deepseek_v4/compress_norm_rope.py)
    *   [compressor.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/kernels/deepseek_v4/compressor.py)
    *   [mla.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/kernels/deepseek_v4/mla.py) (with the `sinks` broadcasting and online softmax NaN fixes)
    *   [mla_swa.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/kernels/deepseek_v4/mla_swa.py)
    *   [streamindex_topk.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/kernels/deepseek_v4/streamindex_topk.py)
    *   [test_compress_norm_rope.py](file:///Users/patemotter/dsv4/torchtpu-vllm/tests/kernels/test_compress_norm_rope.py)
    *   [test_dsv4_mla.py](file:///Users/patemotter/dsv4/torchtpu-vllm/tests/kernels/test_dsv4_mla.py)
    *   [test_dsv4_mla_nan.py](file:///Users/patemotter/dsv4/torchtpu-vllm/tests/kernels/test_dsv4_mla_nan.py)
*   **Verification:** All unit tests pass successfully on the TPU VM.

### Commit 2: Implement PyTorch Custom Op Wrappers [STATUS: COMPLETED]
*   **Commit Message:** `feat(deepseek_v4): implement PyTorch custom op wrappers`
*   **Files Added/Modified:**
    *   `src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py` (PyTorch custom op wrappers and tracing fakes)
    *   [sharding.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/layers/common/sharding.py) (Added `BATCH`, `CONTEXT`, `KV_CACHE_HEAD`, and `MODEL` axes)
    *   [attention.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/layers/vllm/attention.py) (Supported `uint8` and FP8 KV cache dtypes)
*   **Verification:** Verified JAX-to-PyTorch tracing and shape registration.

### Commit 3: Implement Model Patching and FP8 Quantization [STATUS: COMPLETED]
*   **Commit Message:** `feat(deepseek_v4): implement model patching and FP8 quantization`
*   **Files Added/Modified:**
    *   `src/vllm_torchtpu/models/vllm/deepseek_v4_patch.py` (Model patching and MHC alignment)
    *   `src/vllm_torchtpu/layers/vllm/quantization/deepseek_v4_fp8.py` (DeepSeek-V4 FP8 quantization class)
    *   `src/vllm_torchtpu/layers/vllm/quantization/fp8.py` (Quantization mapping changes)
    *   `src/vllm_torchtpu/layers/vllm/quantization/__init__.py`
    *   `src/vllm_torchtpu/layers/common/quant_methods.py`
    *   `src/vllm_torchtpu/layers/vllm/moe_routing.py`
*   **Verification:** Verified model loading and weight allocation.

### Commit 4: Register Custom Ops and Platform Configuration [STATUS: COMPLETED]
*   **Commit Message:** `feat(deepseek_v4): register custom ops and platform configuration`
*   **Files Modified:**
    *   [tpu_runner.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/runner/tpu_runner.py) (Registered custom ops and dummy caches)
    *   [tpu_platform.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/platforms/tpu_platform.py) (Registered quantization config and overrode `is_tpu()`)
*   **Verification:** Verified platform initialization.

### Commit 5: Add End-to-End Verification Script [STATUS: COMPLETED]
*   **Commit Message:** `test: add DeepSeek-V4 verification script`
*   **Files Added:**
    *   [verify_dsv4.py](file:///Users/patemotter/dsv4/torchtpu-vllm/examples/verify_dsv4.py)
*   **Verification:** Successfully ran end-to-end inference for the toy model on the TPU VM using `tensor_parallel_size=2` and `gpu_memory_utilization=0.5`.
