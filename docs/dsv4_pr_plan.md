# Incremental PR Plan: DeepSeek-V4 TPU Inference

This plan divides the DeepSeek-V4 TPU integration into 5 logical, sequential Pull Requests (PRs). This allows for methodical review and gives you time to study each component in depth.

---

## Repository 1: `torch_tpu` (Dependency)

Before the vLLM changes can compile, the underlying compiler bridge must support output sharding.

### PR 1: Add Output Sharding Support to `pallas.py`
*   **Objective:** Extend `jax_op` to accept `output_partition_specs` and use them in `torch_placeholder` to construct local-shaped PyTorch fake tensors.
*   **Changes:**
    *   Modify `torch_tpu/_internal/pallas/pallas.py` (the changes currently applied to the virtual environment on the VM).
*   **Study Focus:** Understand how `torch_placeholder` bridges JAX's `ShapeDtypeStruct` back to PyTorch's `FakeTensor` using the sharding specs.

---

## Repository 2: `torchtpu-vllm` (Main Codebase)

Once the `torch_tpu` dependency is updated, you can submit the vLLM integration in 4 sequential PRs.

### PR 2: JAX/Pallas Kernels & Unit Tests
*   **Objective:** Add the raw mathematical JAX/Pallas kernels and their offline unit tests.
*   **Commits:**
    *   `0745b772`: *feat(deepseek_v4): add JAX kernels and unit tests*
*   **Files:** `src/vllm_torchtpu/kernels/deepseek_v4/`
*   **Study Focus:** Review how the JAX/Pallas kernels implement MLA projection and block-sparse indexing on the TPU.

---

### PR 3: PyTorch Custom Op Wrappers & Model Patching
*   **Objective:** Wrap the JAX kernels using `pallas.jax_op` (with the new `output_partition_specs`), implement the PyTorch layer classes, and set up the runtime model patching.
*   **Commits:**
    *   `874a6ec1`: *feat(deepseek_v4): implement PyTorch custom op wrappers*
    *   `6ecb4f5b`: *feat(deepseek_v4): implement model patching and FP8 quantization*
    *   `64031837`: *feat(deepseek_v4): register custom ops and platform configuration*
    *   *Plus the uncommitted changes in [deepseek_v4.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py) (local-shape query view, indexer sharding specs).*
*   **Files:**
    *   `src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py`
    *   `src/vllm_torchtpu/models/vllm/deepseek_v4_patch.py`
    *   `src/vllm_torchtpu/platforms/tpu_platform.py`
    *   `src/vllm_torchtpu/runner/tpu_runner.py`
*   **Study Focus:** Analyze how the local-shape invariant is maintained in the PyTorch graph, and how the `input_partition_specs` and `output_partition_specs` drive sharding propagation.

---

### PR 4: MoE Quantization & JAX-Native Fallback
*   **Objective:** Add MXFP4 MoE quantization, Expert Parallelism (EP=8) sharding, and the JAX-native MoE fallback to prevent VMEM OOM on TPU v6e/v7x.
*   **Commits:**
    *   `73ef8683`: *feat(deepseek_v4): support MXFP4 MoE quantization for full model*
    *   `c4d8d0c2`: *feat(deepseek_v4): enable expert parallelism (EP=8) for full model inference*
    *   `5289c1c5`: *feat(deepseek_v4): patch parallel_state to support EP=8 weight loading and processing*
    *   `d0cf4637`: *feat(deepseek_v4): implement JAX-native MoE fallback to bypass GMM kernel VMEM OOM*
    *   `6fbffe18`: *feat(deepseek_v4): force CPU-side weight processing and bypass GMM prebuild when native MoE is active*
    *   `f36612ce`: *feat(deepseek_v4): update JAX/libtpu deps in pyproject.toml and add VLLM_FORCE_GMM env var for testing*
*   **Files:**
    *   `src/vllm_torchtpu/layers/vllm/quantization/mxfp4.py`
    *   `src/vllm_torchtpu/layers/vllm/fused_moe.py`
    *   `src/vllm_torchtpu/layers/common/fused_moe_gmm.py`
    *   `src/vllm_torchtpu/layers/common/sharding.py`
*   **Study Focus:** Review how the JAX-native fallback uses `jax.shard_map` and standard `matmul` to bypass VMEM allocation limits.

---

### PR 5: Verification & Benchmarking Scripts
*   **Objective:** Add the end-to-end verification and inference scripts to the `examples/` directory.
*   **Commits:**
    *   `0ec82212`: *test: add DeepSeek-V4 verification script*
    *   `2c39614b`: *test: add full model inference script*
*   **Files:**
    *   `examples/verify_dsv4.py`
    *   `examples/run_dsv4_inference.py`
*   **Study Focus:** Use these scripts to verify that future changes do not regress the DeepSeek-V4 compilation path.
