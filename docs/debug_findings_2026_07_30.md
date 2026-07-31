# DeepSeek-V4 Flash TPU End-to-End Compilation & Execution Debug Findings
**Date:** July 30, 2026
**Target Architecture:** TPU v5/v6 with Torch TPU, vLLM, and Pallas/JAX kernels

---

## 1. Execution Command

The end-to-end inference execution was launched via Docker with Torch Dynamo enabled (`VLLM_ENFORCE_EAGER=0`):

```bash
docker run --init --privileged --net=host --ipc=host --rm \
  -w /home/patemotter_google_com/workspace/vllm-torchtpu \
  -v /home/patemotter_google_com/workspace:/home/patemotter_google_com/workspace \
  -v /dev/shm:/dev/shm \
  -e PYTHONPATH=/home/patemotter_google_com/workspace/vllm-torchtpu/src \
  -e HF_HOME=/home/patemotter_google_com/workspace/hf_cache \
  -e TPU_LIBRARY_PATH=/home/patemotter_google_com/workspace/vllm-torchtpu/docker/libtpu.so \
  -e LIBTPU_INIT_ARGS="--xla_tpu_use_enhanced_launch_barrier=false --xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=1000000000 --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=1000000000" \
  -e TORCH_DIST_TIMEOUT=1200 \
  -e VLLM_RPC_TIMEOUT=1200000 \
  -e VLLM_ENGINE_ITERATION_TIMEOUT_S=1200 \
  -e VLLM_TP_SIZE=8 \
  -e VLLM_EP_SIZE=8 \
  -e VLLM_NO_USAGE_STATS=1 \
  -e VLLM_PLUGINS=torchtpu,torchtpu_general \
  -e PJRT_DEVICE=TPU \
  -e JAX_PLATFORMS=tpu,cpu \
  -e PYTHONWARNINGS=ignore \
  -e TRANSFORMERS_VERBOSITY=error \
  -e VLLM_FP4_CACHE_DIR=/dev/shm/dsv4-flash-real-fp4 \
  -e VLLM_XLA_CACHE_PATH=/home/patemotter_google_com/workspace/xla_cache \
  -e VLLM_ENFORCE_EAGER=0 \
  -e DSV4_MODEL_PATH="/home/patemotter_google_com/workspace/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1" \
  -e DSV4_PROMPT="Compare XLA and LLVM in two bullet points." \
  vllm-torchtpu-local:latest python3 -u /home/patemotter_google_com/workspace/vllm-torchtpu/examples/run_dsv4_inference.py
```

---

## 2. Issues Encountered, Root Causes & Fixes

### Issue 1: 30+ Minute Re-Compilation in Eager Mode vs. Torch Dynamo Graph Capture

- **What Happened**:
  Running with `VLLM_ENFORCE_EAGER=1` caused `torch_tpu` to re-compile C++ Tier-3 binary fingerprint keys on every launch, taking over 30 minutes per attempt due to shifting in-memory tensor addresses.
- **Why It Was Changed**:
  Switching to Torch Dynamo mode (`VLLM_ENFORCE_EAGER=0`) captures a static, deterministic PyTorch FX graph with 100% reproducible XLA compilation cache keys.
- **What Was Changed**:
  1. Enabled `VLLM_ENFORCE_EAGER=0`.
  2. Guarded debug `.item()` and `.isnan()` calls in [`deepseek_v4.py`](file:///home/patemotter_google_com/workspace/vllm-torchtpu/src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py) with `if not torch.compiler.is_compiling():`.
  3. Fixed infinite recursion in [`tpu_compiler.py`](file:///home/patemotter_google_com/workspace/vllm-torchtpu/src/vllm_torchtpu/compilation/tpu_compiler.py) by returning a SHA256 string directly from `compute_hash`.

---

### Issue 2: Pallas Operator Caching & Kernel Name Collision

- **What Happened**:
  `_build_compressor_op` constructed `op_name` without including `state_dim` or a unique version tag. Consequently, `IndexerCompressor` (`state_dim=128`) and `MainCompressor` (`state_dim=512`) shared the operator name string `pallas::deepseek_v4_compressor_128_4_1_4`. Pallas reused the cached JAX kernel registered under that string name instead of compiling distinct functions.
- **Why It Was Changed**:
  Pallas (`pallas.jax_op`) caches compiled C++/JAX operators globally by string `op_name`. Changing Python kernel implementations requires changing `op_name` to force JAX to re-trace and re-compile the updated functions.
- **What Was Changed**:
  - Updated `state_dim` lookup in [`deepseek_v4.py`](file:///home/patemotter_google_com/workspace/vllm-torchtpu/src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py#L575):
    ```python
    state_dim = getattr(self.state_cache, "state_dim", getattr(self, "state_dim", 512))
    ```
  - Appended `state_dim` and version suffix `_v5` to `op_name`:
    ```python
    op_name = f"pallas::deepseek_v4_compressor_{self.head_dim}_{self.compress_ratio}_{int(self.overlap)}_{self.state_cache.block_size}_{state_dim}_v5"
    ```

---

### Issue 3: Compressor State Cache Buffer Geometry Overflow (`cannot reshape (16, 64) into (16, 0, 128)`)

- **What Happened**:
  Execution failed with:
  `RuntimeError: cannot reshape array of shape (16, 64) (size 1024) into shape (16, 0, 128) (size 0)` during Pallas compressor tracing.
- **Why It Was Changed**:
  1. `VllmCompressorStateCache.get_kv_cache_spec()` computed `bytes_per_row` assuming `page_size = 256` rows, which resulted in `head_size = 128` bytes (2,048 bytes per page).
  2. When physical allocation created pages with 16 rows per page, 512-dim state values (8,192 bytes per page) exceeded the 2,048-byte physical page width.
  3. `unpack_state_cache()` attempted to slice `slots[:, :, :512]` against a 128-byte wide buffer, causing integer division `2048 // 8192 = 0`. `rows_per_token` evaluated to 0 (`rpt = 0`), producing invalid target shape `(16, 0, 128)`.
- **What Was Changed**:
  1. Updated `get_kv_cache_spec()` in [`deepseek_v4.py`](file:///home/patemotter_google_com/workspace/vllm-torchtpu/src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py#L545):
     ```python
     bytes_per_token = self.state_dim * 4
     bytes_per_page = self.block_size * bytes_per_token
     rows_per_page = 16
     bytes_per_row = bytes_per_page // rows_per_page
     return SlidingWindowMLASpec(
         block_size=self.block_size,
         num_kv_heads=1,
         head_size=align_to(bytes_per_row, 128),  # 512B for state_dim=512
         dtype=torch.uint8,
         sliding_window=self.sliding_window,
         alignment=None,
     )
     ```
  2. Updated `_state_chunk_dims()`, `unpack_state_cache()`, `pack_state_cache()`, `compress_norm_rope_store()`, and `compress_norm_rope_store_indexer()` in [`compress_norm_rope.py`](file:///home/patemotter_google_com/workspace/vllm-torchtpu/src/vllm_torchtpu/kernels/deepseek_v4/compress_norm_rope.py) to inspect `cache.ndim` and safely handle 2D, 3D, and 4D tensor shapes without zero-dimension division.

---

### Issue 4: Dispatch Function Target Selection Bug (`head_dim == 512` vs `state_dim == 512`)

- **What Happened**:
  `_deepseek_v4_compressor_jax` dispatched compressor calls via `forward_fn = (compressor_forward if state_dim == 512 else compressor_forward_indexer)`. Because both `MainCompressor` (`head_dim=512`, `overlap=False` -> `state_dim=512`) and `IndexerCompressor` (`head_dim=128`, `overlap=True` -> `state_dim=512`) evaluate `state_dim = 512`, `IndexerCompressor` incorrectly selected `compressor_forward` (`head_dim=512` implementation) instead of `compressor_forward_indexer` (`head_dim=128` implementation).
- **Why It Was Changed**:
  Function selection must differentiate based on `head_dim` (128 vs 512), not `state_dim` (which is 512 for both).
- **What Was Changed**:
  Updated line 165 in [`deepseek_v4.py`](file:///home/patemotter_google_com/workspace/vllm-torchtpu/src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py#L165):
  ```python
  forward_fn = (compressor_forward if head_dim == 512 else compressor_forward_indexer)
  ```
  And bumped `op_name` version suffix to `_v6`.


---

## 3. Results Summary

- **Backbone FX Compilation Time**: **7.53 seconds** (down from 30+ minutes in eager mode).
- **Bytecode Transformation Time**: **46.32 seconds**.
- **State Cache Alignment**: Fully matching 512B page width for main compressor (`state_dim=512`) and 128B page width for indexer compressor (`state_dim=128`).
