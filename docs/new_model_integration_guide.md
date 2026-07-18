# Developer Guide: Integrating a New JAX-Backed Model into torchtpu-vllm

This guide walks through the mechanical steps and architectural reasons for integrating a new model (like DeepSeek-V4) into the `torchtpu-vllm` engine, using real examples from the codebase.

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

**Real Example (KV Compression Kernel):**
In DeepSeek-V4, the keys and values are compressed before being stored in the cache. The JAX kernel in [compress_norm_rope.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/kernels/deepseek_v4/compress_norm_rope.py) performs RMS Normalization, RoPE embedding, and FP8 quantization in a single fused TPU operation:

```python
# src/vllm_torchtpu/kernels/deepseek_v4/compress_norm_rope.py
import jax
import jax.numpy as jnp

def compress_norm_rope(
    x: jax.Array,                # Input activations: [num_tokens, hidden_size]
    w_norm: jax.Array,           # RMSNorm weight: [hidden_size]
    cos_sin: jax.Array,          # RoPE cos/sin cache: [max_seq_len, rotary_dim]
    positions: jax.Array,        # Token positions: [num_tokens]
    *,
    epsilon: float = 1e-6,
) -> jax.Array:
    # 1. Perform RMS Normalization
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    x_normed = x * jax.lax.rsqrt(variance + epsilon) * w_norm
    
    # 2. Apply RoPE and compress to FP8
    # (Pallas kernel implementation details omitted for brevity)
    ...
    return compressed_fp8_tokens
```

---

### Step 2: Wrap as PyTorch Custom Operators (The Bridge)
PyTorch needs to trace the execution graph before running. Because PyTorch's compiler (Dynamo) cannot trace JAX code, you must wrap the JAX functions and provide a "fake" implementation.

*   **Location:** `src/vllm_torchtpu/layers/vllm/custom_ops/<model_name>.py`
*   **Why:** The wrapper registers the JAX function as a PyTorch operator. The "fake" implementation tells PyTorch what the output shape and dtype will be for any given input shapes, allowing PyTorch to complete its graph tracing.

**Real Example (Compressor Op Wrapper):**
In [deepseek_v4.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py), we wrap the JAX compressor kernel:

```python
# src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py
import torch
from torch_tpu._internal import pallas
from vllm_torchtpu.kernels.deepseek_v4.compress_norm_rope import compress_norm_rope_jax

# 1. Wrap the JAX function as a PyTorch Custom Op
compressor_op = pallas.jax_op("pallas::deepseek_v4_compressor", compress_norm_rope_jax)

# 2. Register the Fake (Meta) Implementation for shape tracing
@compressor_op.register_fake
def _fake_compressor(
    kv_score, ape, norm_weight, cos_sin_cache, positions,
    request_distribution, query_start_loc, seq_lens,
    state_block_tables, cache, *args, **kwargs
):
    # Returns a fake tensor matching the shape and type of the KV cache
    # (No physical memory is allocated on the TPU during tracing)
    return torch.empty_like(cache)

# 3. Implement the PyTorch Module that calls the op
class VllmDeepseekCompressor(torch.nn.Module):
    def forward(self, kv_score, positions, rotary_emb):
        # Retrieve input tensors and metadata...
        # Execute the custom op and copy the output into the pre-allocated cache
        updated_cache = compressor_op(
            kv_score, None, self.norm_weight, cos_sin_cache, positions,
            request_distribution, query_start_loc, seq_lens,
            state_block_tables, self.k_cache.kv_cache
        )
        self.k_cache.kv_cache.copy_(updated_cache)
```

#### Deep-Dive: What does "Fake" mean here?

During the compilation phase (`torch.compile`), PyTorch's compiler (Dynamo) traces the Python code to build a static execution graph. To do this, it passes **`FakeTensor`** objects through the model.

*   **What is a `FakeTensor`?** 
    It is a tensor that contains only **metadata** (`shape`, `dtype`, and `device`), but **no actual data** and **zero physical memory allocation**.
*   **What does the `register_fake` function do?**
    It is a symbolic dry-run function. It must accept the same inputs as the JAX kernel and return a `FakeTensor` representing the output. 
    *   *No Memory Allocation:* You must use functions like `torch.empty(shape, dtype=..., device=...)` or `torch.empty_like(tensor)`. These create the metadata shell without allocating physical High Bandwidth Memory (HBM) on the TPU.
    *   *No Mathematical Execution:* Do not perform any actual math inside the fake function.
*   **Do Shardings need to be defined in the fake?**
    **No.** Shardings (like `jax.sharding.PartitionSpec` or `ShardingAxisName`) are purely JAX-side concepts used by the JAX compiler (`jax.shard_map`) to distribute work across TPU cores. PyTorch's compiler does not understand JAX sharding. The PyTorch fake implementation only needs to return the *logical* shape of the tensor. JAX handles the physical sharding internally when the custom op is executed.

---

### Step 3: Implement Model Patching (The Substitution)
Upstream vLLM defines models using standard PyTorch layers designed for GPUs. Instead of rewriting the entire model, you write a patch script to swap out the GPU layers with your TPU custom operators.

*   **Location:** `src/vllm_torchtpu/models/vllm/<model_name>_patch.py`
*   **Why:** This allows us to reuse 90% of upstream vLLM's model code (weight loading, tokenization, etc.) while replacing only the execution-critical layers (like Attention and MoE) with our TPU-optimized versions.

**Real Example (Attention & SWA Cache Patching):**
In [deepseek_v4_patch.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/models/vllm/deepseek_v4_patch.py), we replace the native AMD/GPU attention and cache modules with our TPU implementations:

```python
# src/vllm_torchtpu/models/vllm/deepseek_v4_patch.py
from vllm_torchtpu.layers.vllm.custom_ops.deepseek_v4 import (
    VllmDeepseekV4MLAAttention, VllmDeepseekV4SWACache
)

def patch_deepseek_v4_model() -> None:
    # 1. Import the original GPU model classes
    import vllm.models.deepseek_v4.amd.model as ds_v4_amd_model
    import vllm.models.deepseek_v4.attention as dsv4_attention

    # 2. Swap them with our TPU subclasses at runtime
    ds_v4_amd_model.DeepseekV4Attention = VllmDeepseekV4MLAAttention
    dsv4_attention.DeepseekV4SWACache = VllmDeepseekV4SWACache
```

---

### Step 4: Define the Quantization Configuration (The Precision)
If the model uses quantized weights (e.g., FP8 for DeepSeek-V4), you must define a quantization class that tells the engine how to process and load these weights.

*   **Location:** `src/vllm_torchtpu/layers/vllm/quantization/<model_name>_fp8.py`
*   **Why:** PyTorch needs to know the layout, block sizes, and scale factors of the quantized weights on disk, and how to allocate the corresponding tensors in TPU memory.

**Real Example (FP8 MoE Quantization Config):**
In [deepseek_v4_fp8.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/layers/vllm/quantization/deepseek_v4_fp8.py), we define how the MoE (Mixture of Experts) weights are loaded and processed. DeepSeek-V4 uses block-wise quantization for experts, which requires custom parameter shapes and scaling factors:

```python
# src/vllm_torchtpu/layers/vllm/quantization/deepseek_v4_fp8.py
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

class DeepseekV4Fp8Config(QuantizationConfig):
    # Specify that we support FP8 activations
    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16]

    # Create the TPU-specific quantization method
    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if layer.__class__.__name__ == "DeepseekV4MoE":
            return VllmFp8MoEMethodTPU(self, prefix)
        return VllmFp8LinearMethodTPU(self, prefix)

class VllmFp8MoEMethodTPU:
    # Custom weight allocation for TPU MoE
    def create_weights(self, layer: torch.nn.Module, ...):
        # Allocate FP8 weights for the experts
        layer.experts.w13_weight = torch.nn.Parameter(
            torch.empty(num_experts, intermediate_size, hidden_size, dtype=torch.float8_e4m3fn),
            requires_grad=False
        )
        # Allocate the inverse scaling factors
        layer.experts.w13_scale_inv = torch.nn.Parameter(
            torch.empty(num_experts, intermediate_size, dtype=torch.float32),
            requires_grad=False
        )
```

---

### Step 5: Register in the Engine (The Hook-Up)
Finally, you must register your new custom operators and quantization configuration in the core engine files.

1.  **Register Quantization:** In `src/vllm_torchtpu/platforms/tpu_platform.py`, add your quantization config to the platform registry.
2.  **Register Custom Ops:** In `src/vllm_torchtpu/runner/tpu_runner.py`, import your patch function and register any custom KV cache allocation sizes.
3.  **Why:** During startup, the engine runs a "warm-up" phase (profiling). It allocates dummy KV caches and executes the model once to force JIT compilation of all JAX kernels. If your ops are not registered here, they will not be compiled ahead of time, leading to compilation latency or OOMs during actual inference.

**Real Example (Platform Registration):**
In [tpu_platform.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/platforms/tpu_platform.py), we register the quantization configuration:

```python
# src/vllm_torchtpu/platforms/tpu_platform.py
class TpuPlatform(Platform):
    def _init_quantization_configs(self):
        # Register the DeepSeek-V4 FP8 config
        from vllm_torchtpu.layers.vllm.quantization.deepseek_v4_fp8 import DeepseekV4Fp8Config
        self.quantization_classes["deepseek_v4_fp8"] = DeepseekV4Fp8Config
```

**Real Example (Runner Op Registration):**
In [tpu_runner.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/runner/tpu_runner.py), we register the custom ops and patch the model class during initialization:

```python
# src/vllm_torchtpu/runner/tpu_runner.py
class TPUModelRunner(GPUModelRunner):
    # 1. Register the custom ops so they are included in the compilation cache key
    _custom_ops = [
        "pallas::deepseek_v4_compressor",
        "pallas::deepseek_v4_indexer",
        "pallas::mla_ragged_paged_attention",
        "pallas::mla_sliding_window_ragged_paged_attention",
    ]

    def load_model(self) -> None:
        # 2. Apply the model class patches before loading weights
        from vllm_torchtpu.models.vllm.deepseek_v4_patch import patch_deepseek_v4_model
        patch_deepseek_v4_model()
        super().load_model()
```

---

## Critical Rules to Avoid Failures

1.  **No Host-Device Syncs in Wrappers:** Inside the `forward` pass of your PyTorch wrappers, never call `.item()`, `.cpu()`, or `.tolist()`. These force the CPU to wait for the TPU, which breaks PyTorch's symbolic graph tracing.
2.  **No JAX Debug Prints in Kernels:** Remove all `jax.debug.print` statements from your JAX kernels before wrapping them. PyTorch Dynamo cannot serialize host callbacks and will crash.
3.  **Allocate KV Caches on TPU:** Ensure all KV caches are allocated on the active TPU device (`self.device`) and support both `uint8` (for SWA) and `FP8` (for MLA) dtypes if required by the model.
4.  **Set Lower Memory Utilization:** When running verification or inference, set `gpu_memory_utilization = 0.5` in the engine arguments. Because the TPU memory allocator reports host-wide totals instead of per-device limits, using the default (`0.9`) will cause the engine to allocate a KV cache that exceeds the physical capacity of a single chip.
