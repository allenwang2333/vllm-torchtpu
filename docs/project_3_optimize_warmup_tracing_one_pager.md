# Design Proposal: Optimize V1 Warmup Tracing (Project 3)

**Author:** Antigravity (Advanced Agentic Coding Partner)  
**Status:** `Proposed`  
**Date:** July 14, 2026  

---

## 1. Problem Statement & Context

In `torchtpu-vllm`, before beginning inference serving, the engine executes a pre-compilation phase known as **warmup** or **capture_model**. During this phase:
-   The runner simulates forward passes across a series of sequence length paddings (e.g. `[16, 32, 64, 128]`).
-   The PyTorch-XLA/Dynamo compiler compiles and caches StableHLO execution graphs on disk for each shape.

### Current Pain Points:
-   **High HBM Footprint:** Currently, the warmup loop allocates distinct dummy key/value caches for each compiled sequence length. These allocations compete for physical HBM with PyTorch parameter weights, frequently triggering `CompileTimeHbmOom` (HBM exhaustion) before compilation completes.
-   **Long Startup Latencies:** Compiling 4 separate static graphs is slow (often taking 4-8 minutes). In containerized GKE deployments, this triggers kubernetes liveness probe timeouts, leading to constant pod restarts.
-   **Lock Inversions:** The difference in compilation speed across different TPU VM ranks during warmup iterations causes worker drift and deadlocks on PJRT device lock collections.

---

## 2. Proposed Solution

We propose implementing **Project 3: Optimize V1 Warmup Tracing**. 
This project introduces two core optimization strategies to minimize warmup memory overhead and accelerate pre-compilation:
1.  **Single-Buffer Slicing:** Pre-allocates a single global memory buffer for the KV cache at the maximum warmup sequence length (`128`). During each warmup iteration, the model forward pass is fed sliced views of this single buffer, allowing the compiler to reuse physical memory locations.
2.  **Dynamic Shapes Compilation:** Enables dynamic shape tracking for the sequence length dimension. Instead of compiling 4 distinct static graphs, the compiler compiles a single symbolic graph, reducing HBM allocation overhead by 4x and startup compilation latency by 75%.

### 2.1 Architectural Flow (Comparison)
```
Current Static Warmup Flow:
  Shape = 16  ──> Allocates cache_16  [16, ...]  ──> Compiles Graph 1 (Static)  ──> HBM used: 1x
  Shape = 32  ──> Allocates cache_32  [32, ...]  ──> Compiles Graph 2 (Static)  ──> HBM used: 2x
  Shape = 64  ──> Allocates cache_64  [64, ...]  ──> Compiles Graph 3 (Static)  ──> HBM used: 3x
  Shape = 128 ──> Allocates cache_128 [128, ...] ──> Compiles Graph 4 (Static)  ──> HBM used: 4x (OOM Risk!)

Proposed Optimized Flow (Single Buffer + Dynamic Shapes):
  Single Allocation: cache_global [128, ...]
  
  Shape = 16  ──> Slices cache_global[:, 0:16]  ──> Compiles Single Dynamic Graph
  Shape = 32  ──> Slices cache_global[:, 0:32]  ──> Reuses Dynamic Graph (No compile!)
  Shape = 64  ──> Slices cache_global[:, 0:64]  ──> Reuses Dynamic Graph (No compile!)
  Shape = 128 ──> Slices cache_global[:, 0:128] ──> Reuses Dynamic Graph (No compile!)
```

---

## 3. Detailed Design

### 3.1 Warmup Loop Buffer Slicing
In `capture_model` inside `vllm_torchtpu/runner/tpu_runner.py`, we replace dynamic loop-level allocations with sliced views of a pre-allocated global buffer:

```python
# Before:
for seq_len in warmup_shapes:
    dummy_kv_cache = torch.zeros((seq_len, ...), device=device)
    model.forward(inputs, dummy_kv_cache)

# After:
global_kv_cache = torch.zeros((max_warmup_len, ...), device=device)
for seq_len in warmup_shapes:
    # Zero-copy view slice
    sliced_kv_cache = global_kv_cache[:, :seq_len]
    model.forward(inputs, sliced_kv_cache)
```

### 3.2 Dynamic Sequence Length Registration
We register the sequence length dimension as dynamic using Dynamo APIs:
```python
# Force Dynamo to treat seq_len dimension as symbolic rather than static
torch._dynamo.mark_dynamic(input_ids, 1)
torch._dynamo.mark_dynamic(positions, 0)
```
This forces the PyTorch Inductor-XLA backend to compile a single StableHLO binary with dynamic dimensions (e.g. `?x512` instead of `16x512`), resulting in a single compilation step that covers all sequence lengths.

### 3.3 Active Memory Purification (Garbage Collection Hook)
To prevent PyTorch's lazy execution accumulation from holding onto intermediate graphs, we insert garbage collection and step barriers inside the warmup loop:
```python
import gc
import torch_xla.core.xla_model as xm

# After each iteration:
xm.mark_step()  # Force execution and free intermediate tracer nodes
gc.collect()    # Reclaim Python memory pools
```

---

## 4. Work Breakdown & Task List

*   **Task 3.1: Refactor Warmup Allocation Flow**
    - Modify `capture_model` in `vllm_torchtpu/runner/tpu_runner.py`.
    - Implement global warmup buffer allocation and slice-based view generation.
*   **Task 3.2: Enable Dynamic Sequence Length Compilation**
    - Configure `TpuCompilerAdaptor` to support dynamic tracing.
    - Register dynamic dimensions on input IDs, positions, and attention masks.
*   **Task 3.3: Compiler Cache Cleanup Hooks Integration**
    - Insert `gc.collect()` and `xm.mark_step()` inside the warmup iterations.
*   **Task 3.4: Validate Compile Time & Memory Reductions**
    - Execute verification scripts and record HBM footprint changes to confirm HBM remains flat during warmup.

---

## 5. Expected System Value & Trade-offs

*   **System Value:**
    - **4x Memory Savings:** Reusing a single buffer avoids multiple concurrent allocations.
    - **75% Faster Warmup:** Compiling a single dynamic graph instead of 4 static graphs reduces engine startup time from ~6 minutes to ~90 seconds.
*   **Trade-offs / Complexity:**
    - **Dynamic Graph Optimization:** Dynamic shape compilation can sometimes result in slightly less optimized StableHLO compared to static dimensions, but the performance difference in attention operations is negligible.
