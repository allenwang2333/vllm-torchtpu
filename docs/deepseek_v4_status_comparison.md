# DeepSeek-V4 TPU Implementation Status & Architectural Comparison

This document provides a detailed comparison of the DeepSeek-V4 (`deepseek-ai/DeepSeek-V4-Flash`) implementation in the PyTorch-based `torchtpu-vllm` repository versus the established JAX-based `tpu-inference` repository.

---

## 1. Executive Summary & Core Paradigm Shift

- **tpu-inference (JAX / torchax):** Operates under a JAX controller as a single global SPMD program, relying on automated mesh partitioning (`shard_map`, `psum`) and compile-time static HBM allocation. Pallas kernels are accessed using `torchax.interop` pointer conversion views (`jax_view` and `torch_view`).
- **torchtpu-vllm (PyTorch / TorchTPU):** Operates as a multi-process executor (`MultiprocExecutor`), running one Python process per physical TPU device/core (MPMD + SPMD) with explicit PyTorch collectives (`tpu_dist`). Pallas kernels are registered as native PyTorch custom operators using `torch_tpu._internal.pallas.jax_op` to allow complete Dynamo tracing (`torch.compile`) without graph breaks.

---

## 2. Detailed Architectural Comparison

| Dimension | JAX Stack (`tpu-inference`) | PyTorch Stack (`torchtpu-vllm`) | Mechanical Impact on DeepSeek-V4 |
| :--- | :--- | :--- | :--- |
| **Distributed Orchestration** | Single global SPMD program; unified JAX controller orchestrating gSMPD/Pathways. | Multi-process executor (`MultiprocExecutor`); one Python process per TPU core. | Collectives cannot rely on JAX mesh auto-partitioning. Ranks must coordinate via explicit `torch.distributed` (`tpu_dist`) calls. |
| **Pallas Kernel Interop** | Pointer conversion views via `torchax.interop` (`jax_view`, `torch_view`). | Traceable custom operators via `torch_tpu._internal.pallas.jax_op`. | Hides JAX kernel internals from PyTorch Dynamo, allowing graph compilation without tracing breaks. |
| **Graph Compilation** | Static JAX JIT compilation (`jax.jit`). | PyTorch Dynamo FX graph tracing (`torch.compile` lowered to StableHLO). | Requires 100% Dynamo fake tensor registration (`register_fake`) for symbolic shape tracing. |
| **Memory Planning** | Eager, compiler-level static HBM allocation for weights and KV caches. | Lazy allocation: `profile_run` executes compiled backbone at peak shape to size HBM before initializing caches. | If `profile_run` memory is inflated by uncompressed weights or view chains, cache initialization fails. |
| **Quantization Execution** | Requantization and sharding run directly on TPU devices via JIT-compiled JAX functions. | Requantization runs on CPU host to avoid TPU HBM allocation spikes; sharded outputs copied to TPU parameters. | Requantization must avoid generating PyTorch view chains that cause compiler memory fragmentation. |

---

## 3. What Has Changed & Why

### 3.1 Pallas Kernel Adaptations for PyTorch/XLA
- **MLIR Signature Specialization:** XLA optimizations (Dead-Code Elimination, Input Deduplication, and Shape-Only optimization) remove unused or redundant operands during compilation, changing the MLIR function signature (e.g. from 14 operands to 8). To prevent signature mismatches with the PyTorch `jax_op` wrapper, signatures were deduplicated, and a dummy sum of all inputs (multiplied by `0`) was added to the kernel outputs to force XLA to preserve them.
- **Strict Squeeze vs. Silent Squeeze:** JAX's `jnp.squeeze` is strict and throws a `ValueError` if the target axis size is not exactly 1. PyTorch's `squeeze` is a silent no-op. Redundant squeezes on tensors that were already squeezed were removed from the JAX wrappers to prevent compilation failures.
- **Lexical Closures:** Helper functions inside JAX/Pallas kernels (e.g. `compute_with_bq` in `mla.py` and `mla_swa.py`) failed to resolve outer-scope variables (such as `num_q_heads`) during compilation, causing `UnboundLocalError`. These variables are now explicitly bound/redefined locally.
- **Attention Sink Slicing:** Corrected sink indexing (`attention_sinks_ref[:num_q_heads]`) to resolve broadcasting mismatches across dynamic query sizes.

### 3.2 Output Projection (`wo_b`) SPMD Reduction
- **Failure:** In JAX, `self.wo_b` executed as a standard `RowParallelLinear` layer because XLA automatically reduced over the sharded contracting dimension (256) of the operands. In PyTorch, calling the linear layer through `jax_op` boundaries broke XLA SPMD auto-reduction, causing 100% NaNs.
- **Fix:** Wrapped `wo_b` in a custom `shard_map` operator (`_wo_b_jax` / `wo_b_op`) to execute the matmul locally on each rank, followed by an explicit `jax.lax.psum` across the TPU model mesh.
- **Subnormal Sanitization:** Sanitized output values (`NaN` and `Inf` to `0.0`) inside the kernel prior to `psum` because the PJRT fast-math all-reduce operations corrupt subnormal or uninitialized padding numbers into NaNs.

### 3.3 MoE Expert Requantization & Disk Serialization Cache
- **Memory Footprint Mitigation:** Uncompressed MoE expert weights in FP8 require ~120 GB of parameter storage, starving the `profile_run` activator of HBM. Weights are requantized to 4-bit (`fp4`) block-512 (50% memory savings).
- **CPU-Side Execution:** Requantization is executed layer-by-layer on host CPU memory, and the results are copied into clean, contiguous row-major TPU device parameters to prevent PyTorch view chain fragmentation (`copy__as_strided_inverse` errors).
- **Garbage Collection:** Explicitly call `del` and `gc.collect()` after each layer's parameter synchronization to purge XLA memory pools.
- **73x Speedup Serialization Cache:** To bypass the slow CPU-bound requantization pass on GKE containers (taking ~64 minutes), a serialization cache was implemented. If `VLLM_FP4_CACHE_DIR` is set, the loader checks for pre-quantized shards matching `rank_{rank}_layer_{layer_idx}.pt`. If found, they are loaded instantly (reducing startup to 52 seconds).

### 3.4 32-Bit Floating-Point Accumulation
- **spelling/Quality Degradation:** JAX accumulated partial dot products in 16-bit `bfloat16`. In a 43-layer model, accumulating 16-bit values compounds truncation errors, shifting `lm_head` argmax selections to orthographically adjacent vocabulary tokens (e.g. spelling errors).
- **Upgrade:** Upgraded accumulation to full 32-bit `float32` across all dense linear operations (`blockwise_kernel.py`) and MoE expert GMM layers (`gmm_v2.py`).

### 3.5 Warmup & Compilation Stability
- **JIT Compiler Deadlock (Lock Inversion):** If rank JIT compilation times diverge (e.g., Rank 0 finishes in <1s while Rank 1 compiles for 75s), faster ranks execute the graph and hold the global PJRT device lock. When the compiling rank finishes, its background thread attempts to call `torch.tpu._clear_cache()`, which tries to acquire the same lock, deadlocking the cluster. Fixed by disabling asynchronous cache clearing in compiler threads.
- **Worker Drift:** Warmup loop iterations are kept in lock-step using explicit `torch.distributed.barrier()` calls to prevent faster ranks from executing subsequent shapes before slower ranks complete compilation.

### 3.6 Output Projection (`_o_proj`) Inverse-RoPE Alignment
- **Discrepancy:** In the JAX reference, `_o_proj` applies `rotary_emb(..., inverse=True)` to the attention output `o` (which contains the trailing Key RoPE block in its last 64 dimensions) before feeding it to the group parallel projection (`wo_a`). In PyTorch (`torchtpu-vllm`), the `_o_proj` custom op override passed the raw attention output directly to `group_proj_op` without reversing the positional rotation, causing a numerical mismatch.
- **Fix:** Restored the `self.mla_attn.rotary_emb(positions, o_f, inverse=True)` call in `VllmDeepseekV4MLAAttention._o_proj` and reshaped the output back to `[t, -1]` before passing to `group_proj_op`.

---


## 4. Parity and Verification Status

Isolated verification scripts were developed to validate the PyTorch wrappers against the reference JAX implementations on physical TPU hardware:

- **Compressor Kernel Parity:** Passed (Maximum absolute difference: `0.0` at mapped active slots).
- **Indexer Kernel Parity:** Passed (Maximum absolute difference: `0` - exact integer match of top-K indices).
- **Attention Kernel Parity:** Passed (Maximum absolute difference: `0.0` for attention output and active SWA cache slots).
- **End-to-End Performance:** Benchmarks run successfully on `alyssa` under `TP=8, EP=8` generating fluent English and python syntax, achieving **13.17 tokens/sec**.

---

## 5. What Is Left (Future Work / Roadmap)

According to the testing audit, the following items remain open:

- **Testing Coverage Gaps:**
  - **MLA Pallas Kernel:** The 50KB JAX kernel handles key/value compression and has 0% direct test coverage in the PyTorch repository. A parameterized unit test suite (`tests/kernels/test_mla.py`) comparing it to a PyTorch reference is planned for Week 4 of the testing roadmap.
  - **Causal Conv1d & All-Gather Matmul Kernels:** Currently untested (0% coverage).
  - **TPUWorker Orchestration:** Lifecycle process setup is untested. Needs mock-based tests to verify import and initialization ordering.
  - **TpuCompilerAdaptor:** Core `compile` caching and FX-to-MLIR graph lowering logic is untested in isolation.
- **Inference Optimization:**
  - **Activation Quantization Noise:** Truncation drift from 8-bit blockwise activation quantization still induces occasional word substitutions. Needs investigation of activation scaling strategies.
  - **EP GMM Block Tuning:** Tuning parameters (`num_kv_pages_per_block` and `num_queries_per_block`) are currently set to `1` (placeholders) in `deepseek_v4.py`.
- **Logistics:**
  - **Production Cache Settings:** Restore `VLLM_FP4_CACHE_DIR` to `"/mnt/local-cache/dsv4-flash-fp4"` inside `dsv4-inference-v7x-8-patemotter-patemotter.yml` to re-enable permanent GCS weight archiving on GKE.
