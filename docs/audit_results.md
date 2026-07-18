# Code Audit: TPU Backend Migration & JAX Elimination

This audit reviews the changes made to the `torchtpu-vllm` repository to eliminate the external `tpu-inference` dependency and minimize the JAX footprint in favor of PyTorch/XLA.

---

## 1. Summary of Changes

We have successfully achieved two major architectural goals:
1.  **Zero External Dependencies:** Removed all imports pointing to the external `tpu-inference` repository. All required kernels and helpers are now in-tree.
2.  **JAX-Free Weight Loading & Orchestration:** Refactored the MXFP4 weight-loading pipeline and the DeepSeek-V4 Indexer query quantization to use pure PyTorch/XLA. JAX is now strictly confined to the execution boundary of the Pallas kernels.

---

## 2. Detailed Audit of Modified Files

### A. Weight Loading & Processing: [mxfp4.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/layers/vllm/quantization/mxfp4.py)
*   **Before:** Used a JAX JIT-compiled function (`process_mxfp4_moe_weights`) that converted PyTorch CPU tensors to JAX CPU arrays via DLPack, performed block-32 to block-512 requantization, sharded them using JAX `NamedSharding`, transposed/padded them, and converted them back to PyTorch/XLA parameters. Required a hack (`os.environ["JAX_PLATFORMS"] = "cpu"`) to prevent JAX from stealing TPU HBM during startup.
*   **After (Pure PyTorch):** 
    *   Dequantization (`dequantize_mxfp4_packed`), padding, block-512 re-quantization (`quantize_tensor_to_fp4`), and packing (`pack_fp4_indices`) are done entirely in **PyTorch on CPU**.
    *   **Local Processing:** Because vLLM already shards the weights on CPU before calling the loader, we process the local shards directly. This completely eliminates the need for JAX `NamedSharding`, `Layout`, and global reordering.
    *   Removed JAX, JAX-DLPack, and `tpu-inference` imports. Removed the `JAX_PLATFORMS=cpu` environment hack.

### B. Custom Ops Wrapper: [deepseek_v4.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py)
*   **Before:** The JAX-compiled indexer wrapper `_deepseek_v4_indexer_jax` performed FP8 query quantization and folded the scales into `indexer_weights` using JAX-based `quantization.quantize_tensor`.
*   **After (PyTorch Boundary):**
    *   Moved the FP8 query quantization (`quantize_tensor`) and scale folding to PyTorch in `VllmDeepseekV4Indexer.forward` (on the PyTorch/XLA side).
    *   The JAX wrapper `_deepseek_v4_indexer_jax` now receives the already-quantized FP8 query and folded weights, passing them directly to the `streamindex_topk` Pallas kernel.
    *   Removed JAX-based `quantization` imports from this file.

### C. JAX Files & Utilities Cleaned
We have removed all JAX helper modules that are no longer needed:
*   **Deleted Directories:**
    *   `src/vllm_torchtpu/layers/common/jax_quantization/` (Deleted - replaced by PyTorch `quantization.py` and PyTorch-side op quantization).
    *   `src/vllm_torchtpu/layers/common/process_weights/` (Deleted - replaced by PyTorch weight processing in `mxfp4.py`).
*   **Deleted Files:**
    *   `src/vllm_torchtpu/layers/common/linear.py` (Deleted).
    *   `src/vllm_torchtpu/layers/common/moe.py` (Deleted).
*   **Reverted Utilities:**
    *   [utils.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/utils.py) and [layers/common/utils.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/layers/common/utils.py) have been **reverted to their original states**. The JAX utilities appended by the subagent (e.g., `device_array`, `DeviceBuffer`, `general_device_put`, `cpu_mesh_context`) were found to be completely unused since we moved weight loading and quantization to PyTorch.

---

## 3. Remaining JAX Footprint Analysis

To run Pallas kernels on TPU, a minimal JAX footprint is still required. The remaining JAX code is strictly categorized below:

### 1. Pallas Kernels (In-Tree)
Located under `src/vllm_torchtpu/kernels/`:
*   `deepseek_v4/`: Compressor (`compressor.py`), Indexer (`streamindex_topk.py`), and helper (`compress_norm_rope.py`).
*   `mla/v2/`: Multi-Head Latent Attention kernels.
*   `fused_moe/v1/`: Megablox GMM and SparseCore ragged gather/reduce kernels.

*Audit Verdict:* **Necessary.** These are the core hardware-optimized kernels. Since Pallas is a JAX-based DSL, these must remain in JAX.

### 2. Custom Op Launches (In-Tree)
Located under `src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py`:
*   Uses `pallas.jax_op` to wrap the Pallas kernels.
*   Uses `jax.shard_map` to distribute kernel execution across the TPU device mesh.

*Audit Verdict:* **Necessary.** This is the bridge that allows PyTorch/XLA to dispatch work to the TPU mesh via JAX.

---

## 4. Verification & Testing Plan

Since we replaced the JAX weight-processing and quantization math with PyTorch, we must verify that:
1.  **No Compilation Failures:** The PyTorch/XLA graph compiles successfully with the new PyTorch-side operations.
2.  **Numerical Parity:** The output of the PyTorch weight-processing matches the JAX version exactly.
3.  **Token Generation:** The model executes and outputs non-empty tokens.

### Verification Steps:
1.  **Recreate Toy Model:** (Currently running - `task-9178`).
2.  **Run 4-Layer Microbenchmark:**
    ```bash
    ssh alyssa "cd ~/workspace/dsv4/torchtpu-vllm && source .venv/bin/activate && HF_HOME=/home/patemotter_google_com/workspace/hf_cache JAX_COMPILER_CACHE_DIR=/home/patemotter_google_com/workspace/jax_cache VLLM_CACHE_ROOT=/home/patemotter_google_com/workspace/vllm_cache PYTHONPATH=~/workspace/dsv4/torchtpu-vllm/src python3 examples/run_dsv4_microbenchmark.py --model /home/patemotter_google_com/workspace/dsv4/torchtpu-vllm/examples/dsv4-flash-4layer --gpu-memory-utilization 0.55 --tensor-parallel-size 8 2>&1"
    ```
    *Check:* Verify that it prints `Generated token IDs` and that they are not all `0` or `PAD` tokens.
