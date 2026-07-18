# Retroactive Design Document: DeepSeek-V4 TPU Inference

This document describes the clean, end-to-end design of the DeepSeek-V4 TPU inference path in `torchtpu-vllm`. 

### Context
DeepSeek-V4 was originally onboarded as a JAX-native model in the `tpu-inference` repository. This design describes the ported architecture in `torchtpu-vllm`, which serves as the unified codebase going forward, wrapping the JAX-native Pallas kernels inside the PyTorch/vLLM frontend. It presents the final, verified architecture from first principles.

---

## 1. System Architecture

The DeepSeek-V4 TPU inference path uses a dual-engine architecture:
1.  **Control Plane (PyTorch):** Handles vLLM engine orchestration, tokenization, KV cache allocation, and weight loading.
2.  **Execution Plane (JAX/Pallas):** Executes the heavy mathematical computations (Attention and MoE) on the TPU using compiled XLA modules.
3.  **Bridge (Pallas Custom Ops):** Connects the two planes using `torch.ops.pallas.jax_op` with explicit sharding propagation.

```
[PyTorch Control Plane]
        |
        v  (torch.compile)
[PyTorch FX Graph] (Local Shapes)
        |
        v  (pallas.jax_op with input/output sharding specs)
[JAX/XLA Compiled Modules] (Global Shapes / ShardMap)
        |
        v
[TPU Hardware Execution]
```

---

## 2. Model Patching Design

To minimize code duplication, we do not define a new model class. Instead, we dynamically patch the upstream vLLM AMD/ROCm model definition during model loading in [tpu_runner.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/runner/tpu_runner.py):

*   **Substituted Classes:**
    *   `vllm.models.deepseek_v4.amd.model.DeepseekV4Attention` -> `VllmDeepseekV4MLAAttention` (TPU)
    *   `vllm.models.deepseek_v4.attention.DeepseekV4SWACache` -> `VllmDeepseekV4SWACache` (TPU)
    *   `vllm.models.deepseek_v4.attention.DeepseekV4Indexer` -> `VllmDeepseekV4Indexer` (TPU)

This allows us to reuse 90% of the upstream vLLM codebase (including weight loading and model topology) while replacing only the execution-critical layers.

---

## 3. Multi-Head Latent Attention (MLA) Layer Design

The MLA layer compresses keys and values into a low-dimensional latent space to minimize KV cache memory footprint. 

### Sharding Strategy
The attention computations are sharded along the head dimension using a 3D JAX Mesh:
*   **Mesh Axes:** `('attn_dp', 'expert', 'model')`
*   **Tensor Parallelism (TP):** Sharded along the `'model'` axis (axis size = TP size, e.g., 8).

### Custom Op Interface
The JAX-native attention kernel is wrapped as a PyTorch custom op `attn_jax_op`:
*   **Inputs:** `q` is sharded in PyTorch with local shape `[t, n_local_heads, head_dim]`.
*   **Input Partition Specs:** `P(None, "model", None)` for `q`, mapping the PyTorch local shape to the JAX global shape.
*   **Output Partition Specs:** `P(None, "model", None)` for the attention output, forcing the JAX compiler to return a local-shaped tensor `[t, n_local_heads, head_dim]` to the PyTorch FX graph, preserving the **Local-Shape Invariant**.

---

## 4. Grouped Projection & Inverse RoPE Design

The output of the attention layer must be projected back to the hidden dimension.

1.  **Local Inverse RoPE:** The inverse Rotary Position Embedding (RoPE) is applied locally on the sharded attention output to avoid cross-device communication.
2.  **Sharded Group Projection:** The group projection `group_proj_op` is sharded and mapped using `shard_map`:
    *   **Input Spec:** `P(None, "model", None)` (receives the local-shaped `[t, 8, 512]` tensor).
    *   **Output Spec:** `P(None, "model")` (returns the local-shaped `[t, 1024]` projection tensor).
    *   **Internal Math:** Inside `shard_map`, the local tensor is reshaped to `[t, 4096]` and multiplied by the local weight shard `[1024, 4096]`.
3.  **Final Reduction:** The local projection is passed to `self.wo_b` (a standard row-parallel linear layer) which performs the final `all-reduce` to sum the outputs across all devices.

---

## 5. Indexer & Compressor Design

For block-sparse attention, the indexer and compressor layers select and compress KV blocks:

*   **Indexer (`indexer_jax_op`):**
    *   **Input Specs:** `q_rope` is sharded `P(None, "model", None)`. `indexer_weights` is a 2D tensor sharded `P(None, "model")`.
    *   **Output Spec:** `P()` (replicated top-k indices).
*   **Compressor (`compressor_jax_op`):**
    *   Since all inputs and outputs are replicated (`P()`), it runs as a replicated op across all devices, updating the local KV cache buffers.

---

## 6. Mixture-of-Experts (MoE) Design

DeepSeek-V4 uses a Mixture-of-Experts (MoE) layer with 256 experts.

### Quantization
*   Supports **MXFP4** block-wise quantization for expert weights to minimize HBM usage.

### JAX-Native Fallback & Pallas GMM
*   To ensure compatibility with the smaller Vector Memory (VMEM) limits of TPU v6e and v7x, the MoE layer implements a JAX-native fallback in [fused_moe_gmm.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/layers/common/fused_moe_gmm.py).
*   It uses `jax.shard_map` and standard `jnp.matmul` instead of the Pallas GMM kernel. The JAX compiler automatically handles memory tiling and paging, avoiding VMEM OOMs while maintaining high performance.

---

## 7. Kernel Verification & Numerical Invariants

To guarantee numerical stability on TPU hardware, the implementation adheres to strict numerical and architectural invariants derived from reference testing against `tpu-inference`:

### A. Softmax Initialization Invariant
*   When accumulating paged attention scores across ragged/masked token sequences in Pallas/TPU, the running max score `m` MUST be initialized to `0.0` (`jnp.zeros`) rather than `-jnp.inf`.
*   **Mechanical Reason:** Initializing to `-jnp.inf` causes `-inf - (-inf) = NaN` whenever an entire block or sequence consists of padding tokens, silently poisoning online softmax accumulation.

### B. Custom Op Sharding Spec Invariant
*   When passing replicated 1D parameter tensors (such as `self.attn_sink`, shape `[64]`) across `pallas.jax_op` into a sharded kernel (such as TP=8, where local query head count is 8), PyTorch XLA does not automatically slice replicated parameters along mesh dimensions.
*   **Mechanical Reason:** You must explicitly slice parameter tensors to their exact local shard dimensions in Python/PyTorch (`local_sink = self.attn_sink[tp_rank * 8 : (tp_rank + 1) * 8]`) before invoking the custom op bridge to prevent broadcasting shape crashes (`(64, 1) vs (8, 128)`).

### C. Standalone Kernel Parity Requirement
*   All underlying Pallas kernels ([mla_swa.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/kernels/deepseek_v4/mla_swa.py), [mla.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/kernels/deepseek_v4/mla.py), [gmm_v2.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/kernels/megablox/gmm_v2.py), and `ragged_gather_reduce`) must be maintained 1-to-1 with the `tpu-inference` reference repository.
*   **Validation Guarantee:** Standalone unit tests (`test_gmm_v2.py` and `test_ragged_gather_reduce.py`) execute in pure JAX without vLLM or PyTorch XLA overhead, guaranteeing 100% mathematical correctness (64/64 tests passing) before integration testing.
