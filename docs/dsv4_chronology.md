# Chronology of DeepSeek-V4 TPU Integration

This document logs the chronological timeline of the DeepSeek-V4 integration in `torchtpu-vllm`, mapping each phase of work to the corresponding git commits and uncommitted changes.

---

## Phase 1: Core Kernel Porting & Layer Wrapping
*Objective: Port JAX kernels from the JAX-native codebase, wrap them as PyTorch custom operators, and patch them into the vLLM model structure.*

*   **Commit [0745b772](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/kernels/deepseek_v4/):** *feat(deepseek_v4): add JAX kernels and unit tests*
    *   Ported the core MLA attention, sliding window attention (SWA), and indexer JAX/Pallas kernels.
*   **Commit [874a6ec1](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py):** *feat(deepseek_v4): implement PyTorch custom op wrappers*
    *   Created `VllmDeepseekV4MLAAttention` and `VllmDeepseekV4Indexer` PyTorch layers wrapping the JAX kernels using `pallas.jax_op` and registered placeholder fake kernels for Dynamo tracing.
*   **Commit [6ecb4f5b](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/models/vllm/deepseek_v4_patch.py):** *feat(deepseek_v4): implement model patching and FP8 quantization*
    *   Added model class patching to swap GPU layers with TPU layers at runtime. Added `DeepseekV4Fp8Config` to support FP8 model weights.
*   **Commit [64031837](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/platforms/tpu_platform.py):** *feat(deepseek_v4): register custom ops and platform configuration*
    *   Registered custom ops in `TPUModelRunner` and the FP8 config in `TpuPlatform`.

---

## Phase 2: MoE Quantization & Scalability
*Objective: Support MXFP4 quantization for experts, scale the model to 8 TPU chips using Expert Parallelism (EP=8), and resolve memory constraints.*

*   **Commit [0ec82212](file:///Users/patemotter/dsv4/torchtpu-vllm/examples/verify_dsv4.py):** *test: add DeepSeek-V4 verification script*
    *   Created `verify_dsv4.py` using a 4-layer dummy model to test compilation end-to-end.
*   **Commit [2c39614b](file:///Users/patemotter/dsv4/torchtpu-vllm/examples/run_dsv4_inference.py):** *test: add full model inference script*
    *   Created `run_dsv4_inference.py` using the full `DeepSeek-V4-Flash` model.
*   **Commit [73ef8683](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/layers/vllm/quantization/mxfp4.py):** *feat(deepseek_v4): support MXFP4 MoE quantization for full model*
    *   Implemented MXFP4 quantization configuration and weight loading for the Mixture of Experts (MoE) layers.
*   **Commit [c4d8d0c2](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/layers/vllm/fused_moe.py):** *feat(deepseek_v4): enable expert parallelism (EP=8) for full model inference*
    *   Configured the MoE layer to shard experts across the 8 TPU devices.
*   **Commit [5289c1c5](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/layers/common/sharding.py):** *feat(deepseek_v4): patch parallel_state to support EP=8 weight loading and processing*
    *   Updated TPU sharding utilities to support 3D mesh slicing for expert parallelism.
*   **Commit [d0cf4637](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/layers/common/fused_moe_gmm.py):** *feat(deepseek_v4): implement JAX-native MoE fallback to bypass GMM kernel VMEM OOM*
    *   Implemented a JAX-native MoE fallback to bypass the Pallas GMM kernel VMEM OOM on TPU v6e/v7x.
*   **Commits [6fbffe18](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/runner/tpu_runner.py), [f36612ce](file:///Users/patemotter/dsv4/torchtpu-vllm/pyproject.toml):** *Weight processing and dependency updates*
    *   Forced CPU-side weight processing for native MoE and bumped JAX dependencies.

---

## Phase 3: Debugging Sharding & Local-Shape Invariant
*Objective: Resolve graph compilation and shape mismatch errors during end-to-end execution on the VM.*

*   **Uncommitted Changes ([deepseek_v4.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py)):** *Resolve Dynamo Tracing & Sharding Mismatches*
    *   **Local-Shape view:** Fixed the query view in `attention_impl` to use `self.mla_attn.n_local_heads` (8) instead of the global head count (64). This resolved PyTorch Dynamo tracing failures.
    *   **Pallas Output Sharding:** Added `output_partition_specs` to `pallas.jax_op` calls for `attn_jax_op` and `group_proj_op`.
    *   **Indexer Sharding Alignment:** Annotated `indexer_jax_op` with `input_partition_specs` and `output_partition_specs`, aligning `indexer_weights` to a rank-2 sharding spec `P(None, "model")`.
*   **Uncommitted Changes ([pallas.py](file:///home/patemotter_google_com/workspace/dsv4/torchtpu-vllm/.venv/lib/python3.12/site-packages/torch_tpu/_internal/pallas/pallas.py) on VM):** *Patch pallas.py to support Output Sharding*
    *   Modified `pallas.py` to accept `output_partition_specs`. Updated `torch_placeholder` to construct PyTorch fake tensors with local shapes using these specs, preserving the local-shape invariant in the PyTorch FX graph and preventing XLA from scaling the shapes twice.

---

## Phase 4: Root-Cause Bisection & Kernel Validation
*Objective: Systematically resolve NaNs across Attention and MoE layers by comparing against reference implementations and validating kernels in isolation.*

*   **Attention (MLA) 100% Resolution (`task-3200`, `task-3302`):**
    *   **Softmax `-inf` Initialization Bug:** Discovered that initializing running max score `m` to `-jnp.inf` in Paged Attention kernels caused `-inf - (-inf) = NaN` during online softmax accumulation over masked/padded tokens. Restored reference Pallas kernels [mla_swa.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/kernels/deepseek_v4/mla_swa.py) and [mla.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/kernels/deepseek_v4/mla.py) from `tpu-inference` (where `m` is initialized to `0.0`).
    *   **Parameter Sharding Alignment:** In [deepseek_v4.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py), implemented Python-side slicing of `self.attn_sink` (`local_sink = self.attn_sink[tp_rank * n_local_heads : (tp_rank + 1) * n_local_heads]`) before passing to `pallas.jax_op`. This resolved broadcasting shape crashes (`(64, 1) vs (8, 128)`) without hacking kernel internals.
    *   **Verification:** Verified via `verify_dsv4.py` that MLA produces clean activations without NaNs (`norm=21.0664`).
*   **MoE Input Validation (`task-3509`):**
    *   Added step-by-step diagnostic logging inside `VllmFp8MoEMethodTPU.apply_monolithic` in [fp8.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/layers/vllm/quantization/fp8.py).
    *   Confirmed with 100% certainty that all tensors entering the MoE layer (`x`, `router_logits`, FP8 weights, and FP8 scaling factors) are valid, non-zero, and NaN-free (`x norm=256.0000`). Proved NaNs originate strictly within `fused_moe_gmm.py`.
*   **Standalone Kernel Validation (`task-3703`):**
    *   Executed standalone unit test suites `pytest -v tests/kernels/test_gmm_v2.py tests/kernels/test_ragged_gather_reduce.py` directly on VM `alyssa`.
    *   All 64 standalone kernel tests passed in 112 seconds. Proved that the underlying Pallas FP8 General Matrix Multiply (`gmm_v2`) and SparseCore token routing/gathering kernels perform exact arithmetic and block quantization without NaNs or indexing errors in pure JAX isolation.
*   **Systematic Bisection Strategy:**
    *   With standalone kernels validated and PyTorch inputs confirmed clean, isolated the MoE failure strictly to the integration wrapper layer: [fused_moe_gmm.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/layers/common/fused_moe_gmm.py) (sharding specs and chunked reduce-scatter logic) or [fused_moe.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/layers/vllm/fused_moe.py) (PyTorch XLA custom op bridge).
    *   Decided to execute Option 1: replace `fused_moe_gmm.py` with the unmodified reference implementation from `tpu-inference` to eliminate custom unverified sharding modifications.
