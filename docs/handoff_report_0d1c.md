# DeepSeek-V4-Flash Integration on Cloud TPU: Handoff & Transition Report

This document provides a comprehensive summary of the engineering work done to integrate and run the **DeepSeek-V4-Flash** model on Cloud TPUs using `torchtpu-vllm`. It is designed to allow a new agent or developer to resume the work with minimal friction.

---

## 1. Project Objective & Context

*   **Goal:** Run end-to-end inference of the `deepseek-ai/DeepSeek-V4-Flash` model on a Cloud TPU VM.
*   **Hardware:** Cloud TPU v7x-8 VM (`alyssa`) containing 8 TPU chips (94.75 GB HBM per chip, 758 GB total HBM, 772 GB host RAM).
*   **Software Stack:** Linux, Python 3.12, PyTorch/XLA, JAX, `torchtpu-vllm` (vLLM v1 TPU runner). The Pallas kernels are currently imported from the external `tpu-inference` repository, but they must be copied in-tree to remove this dependency.
*   **Model Architecture:** DeepSeek-V4-Flash is a hybrid MoE/Dense model featuring:
    *   **Multi-Head Latent Attention (MLA)** with a compression ratio of 4.
    *   **Sliding Window Attention (SWA)** on specific layers.
    *   **Lightning Indexer** (running on SparseCore) for sparse attention routing.
    *   **DeepSeek Compressor** for state compression.
    *   **MXFP4 (float4_e2m1fn)** block-quantized weights for MoE experts.

---

## 2. Architecture & Implementation

The integration maps vLLM's GPU-centric model definition to TPU-optimized JAX/Pallas kernels. The implementation is split across the following key locations:

### Key Files
*   [deepseek_v4.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py): Defines the TPU-compatible custom ops that wrap the JAX/Pallas kernels:
    *   `VllmDeepseekV4MLAAttention`: Wraps the main MLA attention kernel.
    *   `VllmDeepseekV4IndexerCache`: Wraps the SparseCore Indexer cache.
    *   `VllmCompressorStateCache` / `VllmDeepseekCompressor`: Wraps the compressor state cache and projection.
    *   `VllmDeepseekV4SWACache`: Wraps the SWA cache.
*   [tpu_runner.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/runner/tpu_runner.py): Manages model runner initialization, TPU sharding configurations, and KV cache spec collection from the workers.

### Sharding Strategy
*   **Tensor Parallelism (TP=8):** The model is sharded across all 8 TPU chips.
*   **Attention Sharding:** 
    *   The Query projection ($W_Q$) is column-parallel (shards the attention heads).
    *   The Output projection ($W_O$) is row-parallel (handles the `all-reduce` across devices).
    *   The compressed KV representation ($c_t$) and decoupled key ($k_R$) are head-independent and **replicated** across all devices. Thus, the Pallas attention kernels run locally on each device's local cache slice without needing cross-device communication.

---

## 3. The Debugging Journey & Key Resolutions

During the integration, we encountered and resolved several classes of bugs:

### A. KV Cache Allocator & Specification Issues

#### 1. Integer KV Cache Unsupported on TPU
*   **Issue:** The TPU KV cache allocator does not support integer dtypes (`torch.uint8`). SWA and MLA specs defaulted to `torch.uint8` for FP8 storage, causing allocation failures.
*   **Fix:** Updated the specs to return the base config dtype (`torch.bfloat16`) during tracing, except for the Indexer cache which requires `torch.uint8` for a bitcast operation.

#### 2. ZeroDivisionError in HCA Layers
*   **Issue:** The physical storage block size is calculated as `block_size // compress_ratio`. For HCA layers, `compress_ratio = 128`. With the TPU's default block size of 32, this resulted in `32 // 128 = 0`, causing a division-by-zero crash.
*   **Fix:** Capped `compress_ratio` in the allocator specs using `min(self.compress_ratio, block_size)`, ensuring `storage_block_size = 1` for the allocator. The actual kernel execution still uses the original `128`.

#### 3. Nested Submodule Spec Overwrites
*   **Issue:** vLLM collects specs by traversing all submodules. The `VllmDeepseekV4MLAAttention` layer contains a nested base `MLAAttention` module. Both implemented `get_kv_cache_spec`, and the base module's unoptimized spec (with `compress_ratio=128`) was overwriting the parent's capped spec.
*   **Fix:** 
    1.  Updated `tpu_runner.py` to skip the base `MLAAttention` submodule.
    2.  Updated `VllmDeepseekV4MLAAttention.__init__` to dynamically bind the `get_kv_cache_spec` methods of all nested helper modules to the parent's capped method.

---

### B. JAX / Pallas Compilation Issues

#### 4. Lexical Closures in Pallas Kernels
*   **Issue:** Nested helper functions inside the Pallas kernels failed to resolve outer-scope variables (like `num_q_heads`), leading to `UnboundLocalError`.
*   **Fix:** Explicitly extracted and bound these variables locally from the input tensor shapes within the nested functions.

#### 5. MLIR Specialization Mismatches
*   **Issue:** XLA optimizations (Dead-Code Elimination and Input Deduplication) removed or merged unused/identical arguments in the compiled MLIR (e.g., when `swa_only=True` or when passing identical block tables). This caused a signature mismatch with the PyTorch-side custom op wrapper.
*   **Fix:**
    1.  **Deduplication:** Cleaned up the signatures in both PyTorch and JAX to merge identical inputs (e.g., merging `state_block_tables` and `k_block_tables`).
    2.  **DCE Workaround:** Added a dummy sum of all inputs (multiplied by `0`) to the output in the JAX wrappers to force XLA to preserve all operands in the MLIR.

---

### C. The SparseCore Halt & Page Size Unification (Critical Breakthrough)

#### 6. Silent Memory Corruption (Root Cause of the SparseCore Halt)
*   **Issue:** When running the full model, it crashed with a `RuntimeUnexpectedCoreHalt` (SparseCore hardware halt) during warmup.
*   **Discovery:** The main MLA layers had `page_size_padded = None` in their specs. The vLLM allocator fell back to the unpacked page size (**`40,960` bytes**), but the Pallas kernel wrote the packed physical size (**`163,840` bytes**). This caused silent out-of-bounds HBM writes, corrupting memory and halting the TPU.
*   **Fix:** Updated `deepseek_v4.py` to calculate and set the correct `page_size_padded` using `PallasAttentionBackend.get_kv_cache_page_size_bytes(...)` for all MLA and Indexer layers.

#### 7. SWA Spec Unification Page Size Bug
*   **Issue:** With the hybrid KV cache manager disabled on TPU, vLLM's `unify_hybrid_kv_cache_specs` converts all `SlidingWindowMLASpec` (SWA) to `MLAAttentionSpec` and pads their `block_size` to `256`. However, it did not scale `page_size_padded`, causing the compressor state cache (`block_size` padded from 4 to 256) to fail the page-size assertion.
*   **Fix:** Patched `vllm/v1/core/kv_cache_utils.py` on the VM to scale `page_size_padded` proportionally when `block_size` is unified:
    $$\text{new\_page\_size\_padded} = \text{page\_size\_padded} \times \left(\frac{\text{new\_block\_size}}{\text{old\_block\_size}}\right)$$

---

### D. Host Environment Issues

#### 8. Disk Space Exhaustion on Root Partition
*   **Issue:** The TPU VM's root partition (`/dev/root`, 96 GB) became 100% full because the `torch.compile` cache default directory was `/home/patemotter_google_com/.cache/vllm`. This caused compiler writes to fail.
*   **Fix:** Cleared the `vllm` and `uv` caches from the home directory and redirected the cache root to the large workspace partition (`/dev/nvme1n1`, 984 GB) by setting the environment variable:
    `VLLM_CACHE_ROOT=/home/patemotter_google_com/workspace/vllm_cache`

---

## 4. Execution Environment & Launch Commands

To execute runs on the remote TPU VM (`alyssa`), we run commands from the local workspace which syncs to the VM and triggers execution via SSH.

### VM Environment & Paths
*   **SSH Target:** `alyssa` (configured in local `~/.ssh/config`).
*   **Workspace Path (VM):** `~/workspace/dsv4/torchtpu-vllm` (synced from local `/Users/patemotter/dsv4/torchtpu-vllm`).
*   **Virtual Environment (VM):** `~/workspace/dsv4/torchtpu-vllm/.venv` (Python 3.12).
*   **Hugging Face Cache:** `/home/patemotter_google_com/workspace/hf_cache` (redirected via `HF_HOME` to prevent root partition OOM).
*   **JAX Compiler Cache:** `/home/patemotter_google_com/workspace/jax_cache` (redirected via `JAX_COMPILER_CACHE_DIR`).
*   **vLLM Compile Cache:** `/home/patemotter_google_com/workspace/vllm_cache` (redirected via `VLLM_CACHE_ROOT`).
*   **Pallas Kernels Path:** `~/workspace/dsv4/tpu-inference` (added to `PYTHONPATH`).

### Launch Commands

#### 1. Running the Full Inference Model:
```bash
ssh alyssa "cd ~/workspace/dsv4/torchtpu-vllm && source .venv/bin/activate && HF_HOME=/home/patemotter_google_com/workspace/hf_cache JAX_COMPILER_CACHE_DIR=/home/patemotter_google_com/workspace/jax_cache VLLM_CACHE_ROOT=/home/patemotter_google_com/workspace/vllm_cache PYTHONPATH=~/workspace/dsv4/torchtpu-vllm/src:~/workspace/dsv4/tpu-inference python3 examples/run_dsv4_inference.py 2>&1"
```

#### 2. Running the 4-Layer Microbenchmark:
```bash
ssh alyssa "cd ~/workspace/dsv4/torchtpu-vllm && source .venv/bin/activate && HF_HOME=/home/patemotter_google_com/workspace/hf_cache JAX_COMPILER_CACHE_DIR=/home/patemotter_google_com/workspace/jax_cache VLLM_CACHE_ROOT=/home/patemotter_google_com/workspace/vllm_cache PYTHONPATH=~/workspace/dsv4/torchtpu-vllm/src:~/workspace/dsv4/tpu-inference python3 examples/run_dsv4_microbenchmark.py --model /home/patemotter_google_com/workspace/dsv4/torchtpu-vllm/examples/dsv4-flash-4layer --gpu-memory-utilization 0.55 --tensor-parallel-size 8 2>&1"
```

---

## 5. Current Status

### Microbenchmark
*   **Status:** **SUCCESS**
*   **Configuration:** 4-layer model (`dsv4-flash-4layer`) running with `TP=8` on `alyssa`.
*   **Result:** Successfully compiled all graphs, completed the memory profiling and warmup, and executed text generation without any crashes or assertions.

### Full Model Run
*   **Status:** **SUCCESS (Empty Generation Output)**
*   **Configuration:** Full DeepSeek-V4-Flash model running with `TP=8` on `alyssa`.
*   **Result:** The engine initialized, compiled all graphs, completed warmup, and ran generation successfully without any hardware halts or assertions:
    *   `init engine (profile, create kv cache, warmup model) took 3552.66 s (compilation: 1766.56 s)`
    *   `Processed prompts: 100%|██████████| 2/2 [00:30, 15.35s/it, est. speed input: 0.59 toks/s, output: 8.34 toks/s]`
*   **Symptom:** Both prompts completed generation but returned empty strings (`Generated text: ''`). The execution took the full time expected for `max_tokens=128` (30 seconds total for 2 prompts), indicating the model generated tokens (likely pad or EOS tokens repeatedly) rather than stopping immediately.

---

## 5. Next Steps & Remaining Tasks

1.  **Debug the Empty Generation Output:**
    *   *Symptom:* The model runs for the full length of `max_tokens` but outputs empty strings.
    *   *Hypothesis:* The custom Pallas kernels (attention, indexer, or compressor) might be returning zeros or NaNs due to a numerical or layout mismatch under the real weights, causing the model to output pad/EOS tokens.
    *   *Action:* Modify `run_dsv4_inference.py` to print the raw `token_ids` generated by the model. Add debug prints in the custom op wrappers (`deepseek_v4.py`) to inspect the inputs/outputs of the kernels during the forward pass.
2.  **Complete In-Tree Kernel Migration (Eliminate `tpu-inference` Dependency):**
    *   *Status:* The compressor kernel, indexer kernel, and quantization module have **already been copied** from the local `tpu-inference` repo to the local `torchtpu-vllm` workspace under:
        *   [compressor.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/kernels/deepseek_v4/compressor.py)
        *   [streamindex_topk.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/kernels/deepseek_v4/streamindex_topk.py)
        *   [quantization/](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/layers/common/quantization/)
    *   *Action:* Update the imports in `deepseek_v4.py` to use these local paths, sync the codebase to the VM, and remove the `PYTHONPATH` dependency on `tpu-inference`.
3.  **Upstream the vLLM Patch:**
    *   The patch in `/home/patemotter_google_com/workspace/dsv4/vllm/vllm/v1/core/kv_cache_utils.py` (scaling `page_size_padded` in `unify_hybrid_kv_cache_specs`) is currently local to the VM. This should be committed and upstreamed to the project's vLLM fork.
4.  **Clean Up VM Debug Prints:**
    *   Remove the `DEBUG_ASSERTION` print statement from `/home/patemotter_google_com/workspace/dsv4/vllm/vllm/v1/kv_cache_interface.py` on the VM.

