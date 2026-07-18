# DeepSeek v4 TPU Integration: Ramp-Up & Porting Blueprint

This document serves as a self-contained technical reference and structured implementation plan for porting DeepSeek v4 support from the JAX-based `tpu-inference` repository to the native PyTorch-based `torchtpu-vllm` repository.

---

## 1. Architectural Comparison

| Dimension | `tpu-inference` (JAX/torchax) | `torchtpu-vllm` (Native PyTorch) |
| :--- | :--- | :--- |
| **Middle Layer** | JAX / `torchax` interop | Native PyTorch via `torch_xla` |
| **Model Classes** | Custom JAX-based model classes written from scratch | Upstream `vllm` native PyTorch models directly |
| **Pallas Kernel Interop** | Manually mapped using `torchax.interop` (`jax_view` and `torch_view`) | Compiled using `torch_tpu._internal.pallas.jax_op` custom operators |
| **Integration Pattern** | In-tree custom models and registry modifications | Out-Of-Tree (OOT) Platform Plugin with runtime monkey-patching |
| **Graph Compilation** | JAX JIT compilation (`jax.jit`) | PyTorch/Dynamo compilation (`torch.compile(backend="tpu")`) |

---

## 2. End-to-End Execution Flow (torchtpu-vllm)

```
[User Request] 
      │
      ▼
1. Platform Discovery ──► Load TpuPlatform & Apply patches on import
      │
      ▼
2. Worker Spawning    ──► Initialize TPUWorker per chip & Set distributed variables
      │
      ▼
3. Model Loading      ──► Load weights to HBM & Shard across TP/EP axes
      │
      ▼
4. Token Bucketing    ──► Pad input sequences to nearest compile_size (prevents recompiles)
      │
      ▼
5. Graph JIT Compile  ──► (First run) PyTorch-XLA compiles XLA HLO graphs
      │
      ▼
6. Custom Attention   ──► Route forward to Pallas custom attention kernels via jax_op
      │
      ▼
7. Sampling & Output  ──► Samples tokens on host, returns back to client
```

---

## 3. Translation Pattern: torchax vs. jax_op

When porting attention backends and kernels, we must translate the JAX interoperability wrappers.

### Old Pattern (`tpu-inference`)
Uses explicit pointer conversions via `torchax`:
```python
from torchax.interop import jax_view, torch_view

def forward(self, q_tensor, kv_tensor):
    # Convert PyTorch-XLA tensors to JAX views
    q_jax = jax_view(q_tensor)
    kv_jax = jax_view(kv_tensor)
    
    # Run JAX/Pallas computation
    output_jax = jax_pallas_kernel(q_jax, kv_jax)
    
    # Wrap back to PyTorch
    return torch_view(output_jax)
```

### New Pattern (`torchtpu-vllm`)
Exposes the JAX function as a traceable PyTorch custom operator using `torch_tpu`:
```python
from torch_tpu._internal import pallas

# 1. Define the pure JAX function
def jax_kernel_entry(q_jax, kv_jax, *args, **kwargs):
    return jax_pallas_kernel(q_jax, kv_jax)

# 2. Wrap as a custom op (hides JAX internals from Dynamo tracing)
kernel_op = pallas.jax_op(
    "pallas::my_custom_kernel",
    jax_kernel_entry,
    donate_argnums=(0,)
)

# 3. Register a fake implementation for PyTorch tracing
@kernel_op.register_fake
def _fake_op(q, kv, *args, **kwargs):
    return torch.empty_like(q)

# 4. Invoke as a standard PyTorch operator
def forward(self, q_tensor, kv_tensor):
    return kernel_op(q_tensor, kv_tensor)
```

---

## 4. DeepSeek v4 Porting Checklist & Task Board

### Phase 1: Porting Pallas Kernels
* [ ] **Task 1.1:** Create target directory `src/vllm_torchtpu/kernels/experimental/deepseek_v4`.
* [ ] **Task 1.2:** Port pure JAX/Pallas kernels from the existing `tpu-inference` repository to the new target `src/vllm_torchtpu/kernels/experimental/deepseek_v4/` directory:
  * [ ] [compressor.py](file:///Users/patemotter/dsv4/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/compressor.py)
  * [ ] [compress_norm_rope.py](file:///Users/patemotter/dsv4/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/compress_norm_rope.py)
  * [ ] [compress_store.py](file:///Users/patemotter/dsv4/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/compress_store.py)
  * [ ] [mla.py](file:///Users/patemotter/dsv4/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py)
  * [ ] [mla_swa.py](file:///Users/patemotter/dsv4/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py)
* [ ] **Task 1.3:** Create directory `tests/kernels/deepseek_v4` and port unit tests from the existing `tpu-inference` repository:
  * [ ] [compressor_test.py](file:///Users/patemotter/dsv4/tpu-inference/tests/kernels/deepseek_v4/compressor_test.py)
  * [ ] [mla_test.py](file:///Users/patemotter/dsv4/tpu-inference/tests/kernels/deepseek_v4/mla_test.py)
  * [ ] [mla_swa_test.py](file:///Users/patemotter/dsv4/tpu-inference/tests/kernels/deepseek_v4/mla_swa_test.py)

### Phase 2: Implementing PyTorch Custom Ops & Attention Backend
* [ ] **Task 2.1:** Create `src/vllm_torchtpu/layers/vllm/backends/flash_attn_mla.py`.
* [ ] **Task 2.2:** Port `PallasMLAttentionBackend` and `PallasMLAttentionBackendImpl` from the existing [tpu-inference file](file:///Users/patemotter/dsv4/tpu-inference/tpu_inference/layers/vllm/backends/flash_attn_mla.py).
* [ ] **Task 2.3:** Refactor the backend to use `pallas.jax_op` instead of `torchax.interop` views.
* [ ] **Task 2.4:** Register `FLASH_ATTN_MLA` and `PallasMLAttentionBackend` in the existing [tpu_platform.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/platforms/tpu_platform.py) file.

### Phase 3: Quantization & Model Loading Patches
* [ ] **Task 3.1:** Define `DSV4_FP8 = "deepseek_v4_fp8"` in the existing [quant_methods.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/layers/common/quant_methods.py) file.
* [ ] **Task 3.2:** Port `deepseek_v4_fp8.py` quantization config from the existing [tpu-inference config file](file:///Users/patemotter/dsv4/tpu-inference/tpu_inference/layers/vllm/quantization/deepseek_v4_fp8.py) to a new target file `src/vllm_torchtpu/layers/vllm/quantization/deepseek_v4_fp8.py`.
* [ ] **Task 3.3:** Add mapping in `method_to_config` within the existing [__init__.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/layers/vllm/quantization/__init__.py) file.
* [ ] **Task 3.4:** Port `VllmDeepseekV4MLAAttention` and `patch_deepseek_v4_mla_cls()` to a new target file `src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4_attention.py`.
* [ ] **Task 3.5:** Implement `_maybe_patch_for_deepseek_v4` context manager and apply it during `load_model()` in the existing [tpu_runner.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/runner/tpu_runner.py) file.

### Phase 4: KV Cache & Runner Alignment
* [ ] **Task 4.1:** Update `get_kv_cache_spec()` in the existing [tpu_runner.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/runner/tpu_runner.py) file to handle the new `VllmDeepseekV4MLAAttention` class, padding/aligning its head dimensions to 128.
