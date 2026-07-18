# Retroactive Design & Planning Document: DeepSeek-V4 TPU Integration

This document serves as the retroactive planning and design specification for integrating the DeepSeek-V4 model into the `torchtpu-vllm` engine, eliminating the external `tpu-inference` dependency, and verifying execution on Google TPU hardware.

---

## 1. Objective & Scope

The goal of this project was to clean up and refactor the experimental DeepSeek-V4 implementation from the working branch `patemotter-dsv4` onto a clean branch `patemotter-dsv4-clean`, using a granular, verified, and human-like commit history.

### Out of Scope
*   Porting JAX-side stateful layers (`FlaxLinear`, `FlaxMoE`).
*   Modifying the core PyTorch-XLA memory budget calculation logic (resolved via configuration instead).

---

## 2. Background & Requirements

DeepSeek-V4 is a Mixture-of-Experts (MoE) model that introduces two complex attention mechanisms:
1.  **Multi-head Latent Attention (MLA):** Compresses key-value (KV) states into a low-dimensional latent space to reduce KV cache memory.
2.  **Sliding Window Attention (SWA):** Restricts attention to a fixed window of recent tokens to limit cache size.

### System Requirements
*   **Hardware:** Google Cloud TPU VM (e.g., `tpu7x-8` node with 8 chips, 32GB HBM per chip).
*   **Software Stack:** PyTorch, PyTorch-XLA, JAX, Pallas.
*   **Dependency Goal:** Eliminate the external `tpu-inference` package to make `torchtpu-vllm` self-contained.
*   **Stability Goal:** Resolve numerical underflow (`NaN`s) occurring during short-sequence decoding.
*   **Compilation Goal:** Prevent Out-Of-Memory (OOM) failures during the Ahead-Of-Time (AOT) graph compilation phase.

---

## 3. Architectural Design

The integration enforces a strict separation between PyTorch (state and control) and JAX (stateless math execution).

```
+-----------------------------------------------------------------------------+
|                               PyTorch Frontend                              |
|  - Allocates parameters and KV caches.                                      |
|  - Traces the model graph symbolically using Dynamo.                        |
+--------------------------------------+--------------------------------------+
                                       |
                              (torch.ops.pallas)
                                       v
+-----------------------------------------------------------------------------+
|                             JAX/Pallas Backend                              |
|  - Receives memory pointers from PyTorch via DLPack.                        |
|  - Compiles stateless mathematical kernels to XLA.                          |
|  - Writes results directly back to PyTorch memory buffers.                  |
+-----------------------------------------------------------------------------+
```

### Key Design Decisions

#### A. Dead Code Elimination
*   *Decision:* Do not migrate `linear.py`, `moe.py`, or `process_weights/moe_weights.py` from `tpu-inference`.
*   *Rationale:* `torchtpu-vllm` already contains native implementations for FP8 linear layers (`fp8.py`) and GMM-based MoE layers (`fused_moe_gmm.py`). Migrating the JAX-side equivalents would introduce ~1,800 lines of redundant, dead code.

#### B. Softmax Masking (NaN Resolution)
*   *Decision:* Replace additive masking (`logits + mask`, where mask is `-1e9`) with conditional selection (`jnp.where(mask, logits, -jnp.inf)`) in the attention kernels.
*   *Rationale:* In `bfloat16`, adding `-1e9` to small logits causes numerical underflow. The values are lost, and the subsequent exponentiation step ($e^{x-m}$) divides by zero, propagating `NaN`s. Using `jnp.where` compiles to a hardware-level select instruction on the TPU, ensuring masked tokens evaluate to exactly `0.0` in the softmax distribution.

#### C. Memory Budget Allocation
*   *Decision:* Revert custom HBM budget scaling logic in `utils.py` and `tpu_worker.py`. Instead, configure the model runner to use `gpu_memory_utilization = 0.5` and `tensor_parallel_size = 2` (or higher).
*   *Rationale:* `torch.accelerator.get_memory_info` has a hardware limitation where it returns the host-wide total memory of all TPU chips (e.g. 94.75 GB) instead of the local chip capacity (32GB). Attempting to fix this by dividing by `device_count()` is fragile because `device_count()` changes depending on `TPU_VISIBLE_CHIPS` settings. Scaling down the utilization fraction and sharding weights via Tensor Parallelism (TP) is the standard, stable way to keep the memory footprint within physical limits.

---

## 4. Implementation Phases

```
Phase 1: Kernel Migration & Unit Testing
   │
   ▼
Phase 2: Custom Op Wrapping & Tracing Fakes
   │
   ▼
Phase 3: Model Patching & Quantization Setup
   │
   ▼
Phase 4: Platform Hook-up & E2E Verification
```

### Phase 1: Kernel Migration & Unit Testing
*   **Task:** Copy the raw JAX kernels (`compress_norm_rope.py`, `compressor.py`, `mla.py`, `mla_swa.py`, `streamindex_topk.py`) to `src/vllm_torchtpu/kernels/deepseek_v4/`.
*   **Task:** Create isolated unit tests in `tests/kernels/` to verify these kernels against NumPy/CPU references on a single TPU device without JAX meshes.

### Phase 2: Custom Op Wrapping & Tracing Fakes
*   **Task:** Implement `src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py`.
*   **Task:** Wrap JAX functions using `pallas.jax_op` and register `@op.register_fake` functions to return metadata-only `FakeTensor` shapes (`torch.empty` or `torch.empty_like`).
*   **Task:** Support `uint8` KV caches (for SWA) and `FP8` KV caches (for MLA) in the platform's attention allocator.

### Phase 3: Model Patching & Quantization Setup
*   **Task:** Implement `src/vllm_torchtpu/models/vllm/deepseek_v4_patch.py` to replace upstream GPU classes with our TPU subclasses at load time.
*   **Task:** Implement `src/vllm_torchtpu/layers/vllm/quantization/deepseek_v4_fp8.py` to handle the parameter allocation and loading of DeepSeek-V4's FP8 weights.

### Phase 4: Platform Hook-up & E2E Verification
*   **Task:** Register the custom ops in `TPUModelRunner._custom_ops` and the quantization config in `TpuPlatform`.
*   **Task:** Override `is_tpu()` to return `True` in `TpuPlatform` to bypass upstream enum checks during multi-cache binding.
*   **Task:** Write `examples/verify_dsv4.py` and execute it on the TPU VM to verify compilation and generation.

---

## 5. Verification Results

### Standalone Tests
All migrated JAX kernels were verified using standalone tests on the TPU VM:
*   `tests/kernels/test_compress_norm_rope.py` -> **PASSED**
*   `tests/kernels/test_dsv4_mla.py` -> **PASSED**
*   `tests/kernels/test_dsv4_mla_nan.py` -> **PASSED** (Verified that short-sequence logits do not propagate `NaN`s).

### End-to-End Verification
The verification script `examples/verify_dsv4.py` was executed on the `alynie-v7x-8` TPU VM:
*   **Config:** `tensor_parallel_size = 2`, `gpu_memory_utilization = 0.5`, `max_model_len = 128`.
*   **Result:** **SUCCESS**
    *   All JAX and PyTorch graphs (backbone, sampling, custom ops) compiled successfully.
    *   The engine successfully allocated the SWA (`uint8`) and MLA (`FP8`) caches.
    *   The model executed the forward pass and generated tokens without errors.
