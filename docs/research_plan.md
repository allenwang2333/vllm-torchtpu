# DeepSeek V4 Integration: Research & Execution Plan

This document tracks the integration of DeepSeek V4 into `torchtpu-vllm` using PyTorch-native orchestration and JAX/Pallas kernels from `tpu-inference`.

---

## Phase 1: End-to-End Tracing & Graph Verification
*   **Status:** **Completed**
*   **Objectives:**
    *   [x] Integrate the DeepSeek V4 model definition class into the `torchtpu-vllm` loader.
    *   [x] Stub out the custom attention, indexer, and compressor layers.
    *   [x] Resolve all compilation-time and runtime-initialization blockers (SWA cache, TPU integer KV cache limits, HBM budget, and OOT platform checks).
    *   [x] Run the dummy-weight offline inference script on 4 TPUs using Tensor Parallelism to verify a clean HLO graph trace with no Dynamo graph breaks.

## Phase 2: Custom Kernel Integration (`jax_op` Bridge)
*   **Status:** **In Progress**
*   **Objectives:**
    *   [ ] Audit the JAX/Pallas kernels in `tpu-inference` for DeepSeek V4:
        *   MLA attention projection/decoding kernels.
        *   MoE top-k routing and gate kernels.
        *   Fused MoE GEMM kernels.
    *   [ ] Wrap the JAX/Pallas kernels using `torch_tpu.jax_op` to expose them as PyTorch custom operators.
    *   [ ] Implement the `meta` (abstract) tracing registrations for these operators in `torchtpu-vllm` so they can be traced by Dynamo.
    *   [ ] Replace the dummy stubs in [deepseek_v4.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py) with these `jax_op` calls.
    *   [ ] Verify that the model still traces and compiles with the actual operators in place.

## Phase 3: Sharding & Memory Layout Design
*   **Status:** **Pending**
*   **Objectives:**
    *   [ ] Define the parameter sharding strategy for the large MoE layers (expert parallelism vs. tensor parallelism) using `ShardingConfigManager`.
    *   [ ] Align the physical page shapes of the MLA KV cache with the memory layout expectations of the JAX/Pallas attention kernels.
    *   [ ] Implement TPU-specific sharding annotations (`xs.mark_sharding`) for the custom op inputs and outputs.

## Phase 4: Numerical Verification & End-to-End Testing
*   **Status:** **Pending**
*   **Objectives:**
    *   [ ] Run unit tests for each `jax_op` operator using synthetic inputs to verify numerical parity against the original JAX/Pallas implementations.
    *   [ ] Run end-to-end offline inference with dummy weights to verify output sanity.
    *   [ ] Benchmark performance (tokens/sec) and compile times.
