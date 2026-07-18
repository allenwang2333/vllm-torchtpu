# JAX Kernel Migration & Model Integration Guide: DeepSeek-V4 Retrospective

This document serves as a detailed technical guide and retrospective design document for migrating JAX-based custom kernels and integrating JAX-backed models into the PyTorch-frontend `torchtpu-vllm` engine. 

It is written based on the successful integration of the DeepSeek-V4 model, during which we eliminated the external `tpu-inference` dependency, resolved critical numerical instability (`NaN`) issues, and established tracing-compatible JAX-to-PyTorch wrappers.

---

## 1. The JAX-to-PyTorch Architectural Boundary

To maintain a clean, performant, and compilable codebase, we enforce a strict separation of concerns between PyTorch and JAX. 

```
+-----------------------------------------------------------------------------+
|                               PyTorch Frontend                              |
|  - State Management: All weights, KV caches, and activations are PyTorch    |
|    tensors (torch.Tensor) allocated on the TPU device (e.g., xla:0).        |
|  - Model Definition: Subclasses torch.nn.Module and uses standard PyTorch   |
|    layers and execution graph.                                              |
|  - No JAX NNX Layers: flax.nnx.Linear, flax.nnx.Module, etc., are FORBIDDEN. |
+--------------------------------------+--------------------------------------+
                                       |
                              (Dynamo Tracing)
                                       v
+-----------------------------------------------------------------------------+
|                          PyTorch Custom Op Wrapper                          |
|  - Bridges the PyTorch-to-JAX boundary using torch.ops.pallas.              |
|  - Defines symbolic shape tracing via register_fake.                        |
+--------------------------------------+--------------------------------------+
                                       |
                              (Local JAX Execution)
                                       v
+-----------------------------------------------------------------------------+
|                             JAX/Pallas Backend                              |
|  - Mathematical Kernels: Pure JAX/Pallas functions compiled by XLA.         |
|  - Host-Side Weight Processing: JAX-based CPU functions executed during     |
|    load_weights to pack/quantize parameters before TPU upload.              |
+-----------------------------------------------------------------------------+
```

## 1.5 The Dual-Path Quantization Architecture

`torchtpu-vllm` maintains two parallel implementations of quantization and weight-processing utilities:

1.  **PyTorch-Side (`layers/common/quantization.py`):**
    *   *Implementation:* Uses `torch.Tensor` and PyTorch operators.
    *   *Usage:* Used by standard models (Llama, Qwen) to process weights on the CPU host during model loading (`load_weights`). This avoids booting the JAX compiler during weight loading.
2.  **JAX-Side (`layers/common/jax_quantization/`):**
    *   *Implementation:* Uses `jax.numpy` and JAX operators.
    *   *Usage:* 
        *   **Runtime:** Used inside JAX functions that are compiled to TPU custom ops (e.g., dynamically quantizing the query tensor `q` inside the DeepSeek-V4 attention op). Since this execution happens inside the JAX compilation context, it must use JAX-native operators.
        *   **Load Time (DeepSeek-V4 MoE):** Used by `process_weights/moe_weights.py` to handle the highly complex expert-packing and block-wise quantization layouts for DeepSeek-V4, which have not been ported to PyTorch.

### Core Rules
1.  **State is PyTorch; Math is JAX:** JAX is used *only* for stateless mathematical functions (kernels). All state (parameters, optimizer states, KV cache buffers) must be managed as PyTorch tensors.
2.  **No JAX NNX/Flax Layers:** Do not import or use JAX-side stateful modules (like `JaxLinear` or `JaxMoE` from `flax.nnx`). They are dead code in `torchtpu-vllm`. Test and execute JAX-side mathematical functions (like `xla_quantized_matmul` or `fused_moe_func`) directly by passing JAX arrays.

---

### 2. Pure PyTorch Host-Side Weight Processing

To eliminate JAX from the model-loading pipeline, all weight-processing (transposing, packing, and quantization) must be implemented in native PyTorch on the CPU host.

### Translation: JAX Mesh Sharding to PyTorch Slicing
In the JAX-based `tpu-inference` repository, weights are sharded across the TPU mesh devices at load time using JAX SPMD:
```python
# JAX (Deprecated)
ep_sharding = NamedSharding(mesh, P("expert"))
weight = jax.device_put(weight, ep_sharding)
```
In `torchtpu-vllm`, we do not use JAX sharding on the host. Instead, we slice the PyTorch tensor on the CPU host using standard PyTorch slicing, and load the local slice into the parameter:
```python
# PyTorch (Forward-Looking)
# Slice the weight for Expert Parallelism (EP)
expert_slice = weight.chunk(num_experts, dim=0)[local_expert_start:local_expert_end]
# Quantize the local slice using PyTorch utilities
weight_q, weight_scale = quantize_tensor_to_fp8(expert_slice, axis=0)
# Load into PyTorch Parameter
replace_parameter(layer, "weight", torch.nn.Parameter(weight_q, requires_grad=False))
```
The PyTorch XLA runtime automatically transfers the local slice to the correct TPU device.

---

## 3. Runtime Kernel Execution Mechanics

At runtime, PyTorch passes its tensors to the custom op wrapper, which delegates execution to the JAX/Pallas kernel on the TPU.

### The Execution Flow
1.  **Invocation:** The PyTorch layer (e.g., `VllmDeepseekV4MLAAttention`) calls the custom op:
    ```python
    output = self.attn_op(q, kv, sw_cache, ...)
    ```
2.  **DLPack Bridge:** The `torch.ops.pallas` bridge shares the underlying memory pointers of the PyTorch tensors with JAX without copying data.
3.  **Local JIT Compilation:** XLA compiles the JAX/Pallas kernel locally on the TPU chip.
4.  **TPU Execution:** The compiled kernel executes on the TPU, writing the output directly into the pre-allocated PyTorch output tensor.

---

## 4. Deep Technical Deep-Dives & Pitfalls

### Pitfall 1: Numerical Instability in Online Softmax (NaNs)
**The Problem:**
During short-sequence decoding, the attention logits mask contains large negative values (e.g., `-1e9`) for padded tokens. The original JAX MLA kernel implemented masking by adding the mask to the logits:
```python
# UNDERFLOW-PRONE
masked_logits = logits + mask
```
In `bfloat16` (7-bit mantissa, 8-bit exponent), adding `-1e9` to a small logit (e.g., `5.0`) causes immediate underflow. The logit value is completely lost, and the subsequent exponentiation step ($e^{x - m}$) during online softmax results in division by zero, propagating `NaN`s.

**The Solution:**
Perform masking using a conditional select (`jnp.where`) to directly set masked-out positions to `-inf`:
```python
# STABLE
masked_logits = jnp.where(mask, logits, -jnp.inf)
```
The XLA compiler translates this into a hardware-level select instruction. During the exponential pass, $e^{-\inf}$ evaluates to exactly `0.0` in the TPU's vector register, cleanly eliminating the padded tokens.

### Pitfall 2: PyTorch Dynamo Tracing Failures
PyTorch Dynamo compile-traces the JAX code. This process will fail if the code contains host-device boundaries:

1.  **JAX Host Callbacks:** Any `jax.debug.print` statement inside the JAX kernel will raise:
    ```
    NotImplementedError: serialization of host_callbacks is not yet implemented
    ```
    *Rule:* Remove all debug prints from JAX/Pallas kernels before wrapping.
2.  **Eager Host Syncs:** Calling `.item()`, `.cpu()`, or `.tolist()` on a PyTorch tensor inside the forward pass of the wrapper class forces the CPU to block and wait for the TPU, raising:
    ```
    torch._dynamo.exc.Unsupported: GuardOnDataDependentSymNode
    ```
    *Rule:* Keep the forward pass entirely symbolic. Pass runtime values (like sequence lengths or offsets) as tensors rather than Python scalars.

### Pitfall 3: JAX Mesh Partitioning Error
**The Mechanism:**
JAX uses SPMD (`shard_map`, `Mesh`) where the compiler automatically inserts collectives. PyTorch custom ops, however, are executed **locally** on each TPU chip; sharding is managed by PyTorch outside JAX.

If you run a unit test for a JAX custom op (containing Pallas kernels) under a JAX `Mesh`, JAX will attempt to partition the Pallas kernel and fail:
```
NotImplementedError: Mosaic kernels cannot be automatically partitioned. Please wrap the call in a shard_map.
```
Pallas/Mosaic kernels write directly to TPU memory and cannot be partitioned by the JAX compiler.

**The Solution:**
Run unit tests on a single local device without JAX meshes. To test Expert Parallelism (EP) paths, manually slice the input tensors in Python to simulate sharded inputs, pass them to the local JAX function, and verify the local outputs (as implemented in [test_moe.py](file:///Users/patemotter/dsv4/torchtpu-vllm/tests/layers/common/test_moe.py#L63-L163)).

### Pitfall 4: Quantization Reduction Axis & Broadcasting
When quantizing a weight matrix $W$ of shape `[K, N]` (input features `K`, output channels `N`) to FP8:
*   To get per-channel quantization (one scale factor per output channel `N`), you must reduce over the contracting dimension `K` (axis 0).
*   **Correct Code:**
    ```python
    w_q, w_scale = quantize_tensor(jnp.float8_e4m3fn, w, axis=0)
    # w_scale shape: [N]
    ```

In `xla_quantized_batched_matmul`, the scale is broadcast to the output shape `[B, M, N]` (where `B` is batch, `M` is tokens, `N` is channels) by prepending `1`s:
```python
n_leading = out.ndim - w_scale.ndim  # 3 - 1 = 2
for _ in range(n_leading):
    w_scale = jnp.expand_dims(w_scale, 0)  # scale shape becomes [1, 1, N]
```
This broadcasts perfectly. However, if the weight is batched (`[B, K, N]`) and quantized per-expert, `w_scale` will be `[B, N]` (`ndim=2`). The utility will prepend one `1`, resulting in `[1, B, N]`, which fails to broadcast with `[B, M, N]`. 

*   *Rule:* For batched weights, manually reshape `w_scale` to `[B, 1, N]` before performing the multiplication.

### Pitfall 5: OOT TPU Platform Verification
**The Problem:**
In vLLM, out-of-tree (OOT) platforms like `TpuPlatform` are registered dynamically. During registration, its `_enum` is set to `PlatformEnum.OOT` instead of `PlatformEnum.TPU`. Upstream vLLM check methods like `is_tpu()` check `self._enum == PlatformEnum.TPU`, which returns `False`.
For models like DeepSeek-V4 that use multiple KV cache types (SWA + MLA) in the same layer, this triggers a `NotImplementedError` in `bind_kv_cache` because vLLM thinks it is running on a non-TPU platform that does not support multi-cache binding.

**The Solution:**
Explicitly override `is_tpu(self) -> bool` to return `True` in `TpuPlatform` in [tpu_platform.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/platforms/tpu_platform.py) to bypass the enum check.

### Pitfall 6: KV Cache Dtype Support
**The Problem:**
DeepSeek-V4 uses Sliding Window Attention (SWA) on some layers and Multi-head Latent Attention (MLA) on others. SWA uses a `uint8` cache format to store quantized keys/values, whereas MLA uses `FP8` (`float8_e4m3fn`).
If the platform's attention backend or allocator does not explicitly support `uint8` or `FP8` KV cache dtypes, it will fail during cache initialization.

**The Solution:**
1.  Update `_resolve_kv_cache_dtype` in [attention.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/layers/vllm/attention.py) to resolve `fp8` to the concrete platform FP8 type (`torch.float8_e4m3fn` or `torch.float8_e5m2`).
2.  Update `PallasAttentionBackend.get_kv_cache_shape` to allow `torch.uint8` cache dtypes.

### Pitfall 7: Memory Allocation and TP Scaling
**The Problem:**
On TPU, `torch.accelerator.get_memory_info` has a known limitation where it returns the total memory of the entire PJRT topology (e.g. 94.75 GB for 3 active chips) instead of the per-device memory (32GB), regardless of the device index or `TPU_VISIBLE_CHIPS` setting.
If this total is used directly, vLLM will overestimate the local device capacity and try to allocate a massive KV cache, leading to `RuntimeBufferAllocationFailure` (OOM).

**The Solution:**
Instead of writing complex, non-standard memory-scaling logic in the library code, align with the established pattern on the working branch:
1.  Keep the library memory code clean and identical to the original branch.
2.  In the runner or serving configuration, set `gpu_memory_utilization = 0.5` to scale down the target cache allocation.
3.  Use Tensor Parallelism (`tensor_parallel_size = 2` or `8`) to distribute the model weights across multiple chips, reducing the per-chip weight memory and leaving enough headroom for compilation and the KV cache.

---

## 5. Step-by-Step Integration Workflow

When integrating a new JAX-backed model or migrating JAX kernels, follow this structured workflow:

### Step 1: Code Localization
Copy only the JAX mathematical functions and kernels. **Do not copy JAX NNX/Flax layers, and do not copy JAX weight-processing files.**
*   Kernels: `src/vllm_torchtpu/kernels/<model_name>/`

Run a global search-and-replace to update all imports:
```python
# Replace
from tpu_inference.kernels.experimental.deepseek_v4 import mla
# With
from vllm_torchtpu.kernels.deepseek_v4 import mla
```
Verify that a grep search for the old package name returns zero active imports.

### Step 2: Standalone Kernel Verification
Before writing PyTorch wrappers, write a standalone JAX script (e.g., in `tests/kernels/`) to execute the JAX kernel directly on a single TPU device. 
*   Feed it dummy inputs and compare the output against a simple NumPy/CPU reference.
*   Do not run the standalone test under a JAX `Mesh`.

### Step 3: PyTorch Custom Op Wrapping
Wrap the JAX function using `pallas.jax_op` to register it as a PyTorch custom operator.
```python
from torch_tpu._internal import pallas

# Define the JAX function
def J_op(x, w):
    return jnp.matmul(x, w)

# Wrap as PyTorch custom op
py_op = pallas.jax_op("pallas::my_custom_op", J_op)

# Register a fake implementation for shape tracing
def _fake_op(x, w):
    return torch.empty(x.shape[0], w.shape[1], dtype=x.dtype, device=x.device)
py_op.register_fake(_fake_op)
```

### Step 4: End-to-End Tracing & Toy Model Validation
To verify that PyTorch Dynamo can compile the entire graph without OOM or tracing errors, create a scaled-down "toy" version of the model (e.g., 4 layers, small hidden size).
Run the model using the vLLM API with conservative memory settings:
```bash
# Limit KV cache pre-allocation and sequence length to prevent OOM during profiling
python3 run_dsv4.py --gpu-memory-utilization=0.3 --max-model-len=2048
```

### Step 5: Port and Simplify Unit Tests
Migrate JAX-side unit tests to `tests/layers/common/` using `absl.testing`.
*   **Avoid JAX Layers:** Test the JAX-side mathematical functions (like `xla_quantized_matmul` or `fused_moe_func`) directly by passing JAX arrays, rather than wrapping them in NNX modules.
*   **Avoid Weight Loaders:** Eliminate any dependency on complex weight-loading utilities (like `JaxAutoWeightsLoader`). Generate inputs and weights as JAX/NumPy arrays directly in the test.
*   **Model Quantization Noise:** Use realistic tolerances (`rtol=0.2, atol=0.8` for FP8 matmuls, and `rtol=1e-2, atol=0.15` for BF16 MoE kernels) to account for expected quantization noise.

### Step 6: Implement Host-Side Weight Processing in PyTorch
Instead of copying JAX weight loaders, implement the CPU-host weight processing and quantization in PyTorch inside `layers/vllm/quantization/` (e.g. `fp8.py` or `mxfp4.py`).
*   Slice the weights along the TP/EP dimensions using PyTorch's `.chunk()`.
*   Apply FP8/FP4 quantization using the PyTorch-side `layers/common/quantization.py` utilities.
*   Load the resulting PyTorch tensors directly into the model parameters.
