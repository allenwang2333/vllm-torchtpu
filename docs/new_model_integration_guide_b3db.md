# Developer Guide: Integrating a New JAX-Backed Model into torchtpu-vllm

This guide walks through the mechanical steps and architectural reasons for porting JAX-native Pallas kernels from the legacy `tpu-inference` repository and integrating them into the PyTorch-based `torchtpu-vllm` engine. 

Going forward, `torchtpu-vllm` serves as the unified TPU inference codebase, replacing `tpu-inference` by wrapping high-performance JAX/Pallas kernels inside the PyTorch/vLLM ecosystem.

---

## The Core Architecture: PyTorch to JAX Boundary

`torchtpu-vllm` uses a dual-engine design to run models on Google TPUs:
*   **PyTorch (Frontend/Control):** Manages the model structure, loads weights, allocates memory (including KV caches), and orchestrates the execution loop.
*   **JAX/Pallas (Backend/Execution):** Executes the heavy mathematical computations (like attention and mixture-of-experts) directly on the TPU hardware using highly optimized kernels.

To connect these two, we use a bridge called `torch.ops.pallas.jax_op`. This allows PyTorch to pass memory pointers directly to JAX, which compiles and executes the math on the TPU, writing results back into PyTorch memory without copying data.

---

## Vocabulary & Jargon

*   **Kernel:** A stateless, highly optimized function that runs directly on the TPU hardware.
*   **Dynamo Tracing:** The process where PyTorch's compiler analyzes Python code to construct a static execution graph before running it.
*   **Fake Tensor (Meta Tensor):** A placeholder tensor that has a shape and data type (dtype) but contains no actual data. Used by PyTorch during tracing to calculate output shapes.
*   **Quantization:** Converting high-precision numbers (like 16-bit floats) to lower-precision numbers (like 8-bit floats or integers) to reduce memory usage.
*   **Sliding Window Attention (SWA):** An attention mechanism that only keeps track of a fixed window of recent tokens to save memory.
*   **Multi-Head Latent Attention (MLA):** An attention mechanism that compresses keys and values into a low-dimensional latent space to reduce KV cache size.
*   **Local Shape:** The shape of a tensor on a single TPU device (e.g. `[batch, local_heads, head_dim]`).
*   **Global Shape:** The logical shape of the tensor across the entire TPU mesh (e.g. `[batch, global_heads, head_dim]`).

---

## The Local-Shape Invariant & Sharding

When running on a sharded TPU mesh (e.g., using Tensor Parallelism), `torchtpu-vllm` enforces a strict **Local-Shape Invariant** in the PyTorch FX graph:

1.  **PyTorch Tracing Invariant:**
    *   During `torch.compile`, PyTorch's Dynamo traces the model. Because the physical execution is sharded across devices, all tensors in the PyTorch graph representing sharded data (like attention queries or outputs) **must maintain their local shapes**.
    *   Any PyTorch operations (like `view` or `reshape`) performed on these tensors must use the local dimensions (e.g., viewing queries with `n_local_heads` instead of `n_heads`). Using global dimensions will cause shape mismatches and compilation failures.

2.  **JAX Sharding Propagation:**
    *   JAX operates on **global shapes** internally. When a custom op is executed, the JAX backend compiler converts the PyTorch local-shaped tensors to JAX global-shaped arrays using the `input_partition_specs` provided to `pallas.jax_op`.
    *   If an input tensor is sharded along the `'model'` axis (e.g. with spec `P(None, "model", None)`), the JAX backend compiler scales the sharded dimension of the PyTorch local shape by the size of the `'model'` mesh axis to reconstruct the JAX global shape.

3.  **The Output Sharding Bridge:**
    *   By default, `jax.export.export` (used to compile JAX functions to MLIR) treats the export boundary as replicated (`P()`), stripping the sharding information from the output.
    *   This causes the JAX compiler's `torch_placeholder` to return a global-shaped tensor to PyTorch.
    *   To maintain the local-shape invariant, `pallas.jax_op` supports `output_partition_specs`. When specified, these specs override the stripped JAX export sharding, forcing the JAX compiler to return a local-shaped PyTorch tensor.

---

## Step-by-Step Integration Workflow

To add a new model, you must implement and register five distinct components.

```
[1. JAX Kernels] ---> [2. PyTorch Custom Ops] ---> [3. Model Patching]
                             |                              |
                             v                              v
                      [5. Runner Reg] <--------------- [4. Quant Config]
```

---

### Step 1: Write the JAX Kernels (The Math)
You must write the raw mathematical operations in JAX/Pallas. These functions must be completely stateless (no internal variables or weights; everything must be passed as inputs).

*   **Location:** `src/vllm_torchtpu/kernels/<model_name>/`
*   **Why:** TPUs require compiled XLA (Accelerated Linear Algebra) instructions to execute. We write these operations in JAX because JAX compiles directly to highly efficient XLA code.

---

### Step 2: Wrap as PyTorch Custom Operators (The Bridge)
PyTorch needs to trace the execution graph before running. Wrap the JAX functions using `pallas.jax_op`, specifying both input and output sharding partition specs.

*   **Location:** `src/vllm_torchtpu/layers/vllm/custom_ops/<model_name>.py`
*   **Why:** The wrapper registers the JAX function as a PyTorch operator. Specifying `input_partition_specs` and `output_partition_specs` ensures the JAX compiler correctly transitions between PyTorch's local shapes and JAX's global shapes.

**Example (Sharded Attention Op):**
```python
# src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py
import torch
from torch_tpu._internal import pallas
from jax.sharding import PartitionSpec as P

attn_jax_op = pallas.jax_op(
    "pallas::deepseek_v4_attention",
    _deepseek_v4_attention_jax,
    mesh=mesh,
    input_partition_specs=(
        P(None, "model", None),  # q (sharded along head dim)
        P(),                     # new_kv (replicated)
        ...
    ),
    output_partition_specs=(
        P(None, "model", None),  # out (sharded along head dim)
        P(),                     # new_cache (replicated)
    )
)
```

---

### Step 3: Implement Model Patching (The Substitution)
Upstream vLLM defines models using standard PyTorch layers designed for GPUs. Write a patch script to swap out the GPU layers with your TPU custom operators.

*   **Location:** `src/vllm_torchtpu/models/vllm/<model_name>_patch.py`
*   **Why:** This allows us to reuse 90% of upstream vLLM's model code (weight loading, tokenization, etc.) while replacing only the execution-critical layers (like Attention and MoE) with our TPU-optimized versions.

---

### Step 4: Define the Quantization Configuration (The Precision)
If the model uses quantized weights (e.g., FP8 for DeepSeek-V4), define a quantization class that tells the engine how to process and load these weights.

*   **Location:** `src/vllm_torchtpu/layers/vllm/quantization/<model_name>_fp8.py`

---

### Step 5: Register in the Engine (The Hook-Up)
Finally, register your new custom operators and quantization configuration in the core platform and runner files:

1.  **Register Quantization:** In `src/vllm_torchtpu/platforms/tpu_platform.py`.
2.  **Register Custom Ops:** In `src/vllm_torchtpu/runner/tpu_runner.py`.

---

## Critical Rules to Avoid Failures

1.  **Preserve the Local-Shape Invariant:** Always view and reshape sharded tensors in PyTorch using local dimensions (e.g. `n_local_heads`).
2.  **Annotate Output Shardings:** Always provide `output_partition_specs` for sharded custom JAX ops to prevent JAX-export from returning global-shaped tensors to the PyTorch graph.
3.  **Match Sharding Spec Ranks:** Ensure the rank of the `PartitionSpec` matches the rank of the tensor (e.g., use `P(None, "model")` for a 2D tensor, not `P(None, "model", None)`).
4.  **No Host-Device Syncs in Wrappers:** Never call `.item()`, `.cpu()`, or `.tolist()` in the PyTorch layer forward pass.
