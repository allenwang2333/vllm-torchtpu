# Understanding the Architectural Transition: PyTorch/XLA vs. JAX serving

This document provides a deep, mechanical breakdown of the three core engineering pillars behind our transition from the JAX-based `tpu-inference` repository to the native PyTorch-based `torchtpu-vllm` repository. 

---

## Pillar 1: The PyTorch-to-JAX `torchax` Translation Layer

### 1. Concept
`torchax` is a compiler framework that translates a PyTorch `nn.Module` computational graph into a JAX functional program, allowing PyTorch models to execute inside JAX serving engines.

### 2. Walkthrough
*   **Graph Extraction:** When a model is loaded in `tpu-inference`, `torchax` inspects the PyTorch module's parameters (weights) and its `forward()` method.
*   **Operator Mapping:** It maps each PyTorch operator (e.g., `torch.matmul`, `torch.nn.functional.rms_norm`) to its equivalent JAX operator (e.g., `jnp.matmul`, `jax.nn.rms_norm`).
*   **Function Compilation:** It compiles the entire forward pass into a pure JAX function: `jax_forward_fn(params, inputs, cache)`.
*   **JIT Loop Execution:** The JAX-based serving engine ([decode_loop.py](../../tpu-inference/tpu_inference/runner/decode_loop.py)) wraps this compiled function inside a JAX JIT compilation block (`jax.jit(decode_step)`).
*   **HBM Layout:** During execution, the model parameters and KV cache are stored strictly as JAX arrays in TPU High Bandwidth Memory (HBM).

### 3. Pseudocode
```python
# 1. Instantiate the standard PyTorch model
pytorch_model = VLLMLlamaForCausalLM(config)

# 2. Compile to JAX via the torchax translation layer
jax_params, jax_forward_fn = torchax.compile(pytorch_model)

# 3. Execute inside JAX JIT compilation serving loop
@jax.jit
def decode_step(params, inputs, cache):
    logits, new_cache = jax_forward_fn(params, inputs, cache)
    next_token = jax.lax.argmax(logits, axis=-1)
    return next_token, new_cache
```

### 4. Real Code (Minimal Simulation)
```python
import torch
import torch.nn as nn
import jax
import jax.numpy as jnp

# A simple PyTorch layer
class TorchLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4, 4))
    def forward(self, x):
        return torch.matmul(x, self.weight)

# Simulation of the torchax compilation process
def compile_torch_to_jax(torch_module: nn.Module):
    # 1. Extract parameters and convert to JAX arrays
    jax_params = {
        name: jnp.array(param.detach().numpy())
        for name, param in torch_module.named_parameters()
    }
    
    # 2. Map the forward graph to JAX operations
    def jax_forward_fn(params, x):
        # Maps torch.matmul -> jnp.matmul
        return jnp.matmul(x, params["weight"])
        
    return jax_params, jax_forward_fn

# Run the simulation
torch_layer = TorchLinear()
jax_params, jax_forward = compile_torch_to_jax(torch_layer)
x_jax = jnp.ones((1, 4))

# Compile and execute the JAX function
compiled_step = jax.jit(jax_forward)
out = compiled_step(jax_params, x_jax)
print("JAX Execution Output:", out)
```

---

## Pillar 2: The JAX-to-PyTorch Pointer Bridge (`torchax.interop` vs. `pallas.jax_op`)

### 1. Concept
A pointer bridge is a mechanism that allows JAX/Pallas custom kernels (written in JAX) to run directly on memory buffers allocated by the PyTorch-XLA runtime without copying the underlying data.

### 2. Walkthrough
*   **The Original `torchax` Path (`tpu-inference`):**
    *   The overall graph and loop execute under JAX.
    *   PyTorch-XLA tensors are cast to JAX arrays using `torchax.interop.jax_view(tensor)`. This retrieves the physical PJRT memory pointer of the tensor and wraps it as a `jax.Array` object.
    *   The JAX/Pallas kernel executes on this `jax.Array`.
    *   The output `jax.Array` is wrapped back to a PyTorch-XLA tensor using `torch_view(array)`.
    *   **The Catch:** This manual casting is opaque to PyTorch's Dynamo compiler, causing immediate graph breaks and preventing unified compilation.
*   **The New `jax_op` Path (`torchtpu-vllm`):**
    *   The overall graph and loop execute under PyTorch-XLA.
    *   The Pallas kernel is defined as a pure JAX function.
    *   We wrap the JAX function using [torch_tpu._internal.pallas.jax_op](../src/vllm_torchtpu/layers/vllm/attention.py#L9).
    *   Under the hood, `jax_op` intercepts PyTorch-XLA tensors, extracts their physical PJRT memory pointers, passes them to the JAX compiler/runtime to execute the Pallas kernel, and returns the output memory buffer wrapped as a PyTorch-XLA tensor.
    *   **The Benefit:** It registers the operation as a standard PyTorch custom operator with registered fake implementations. Dynamo traces it cleanly as a single static node in the graph, permitting PyTorch-XLA to compile the entire forward pass as a single, continuous, optimized HLO graph.

### 3. Pseudocode
```python
# Old Path (torchax): Manual, opaque pointer casting inside JAX loop
def forward_torchax(pytorch_q, pytorch_kv):
    jax_q = jax_view(pytorch_q)
    jax_kv = jax_view(pytorch_kv)
    jax_out = pallas_attention_kernel(jax_q, jax_kv)
    return torch_view(jax_out)  # Causes Dynamo graph break

# New Path (jax_op): Traceable PyTorch custom operator inside PyTorch loop
attention_op = pallas.jax_op(
    "pallas::attention",
    jax_pallas_attention_entrypoint,
    donate_argnums=(0,)
)
# Dynamo traces this cleanly; compiles into a single unified HLO graph
def forward_native(pytorch_q, pytorch_kv):
    return attention_op(pytorch_q, pytorch_kv)
```

### 4. Real Code (Minimal Simulation)
```python
import torch
import torch_xla.core.xla_model as xm
from torch_tpu._internal import pallas

# 1. Define the functional JAX/Pallas operation
def jax_scale_array(x, factor):
    import jax.numpy as jnp
    return x * factor

# 2. Wrap as a PyTorch custom operator using the jax_op bridge
scale_op = pallas.jax_op(
    "pallas::scale_op",
    jax_scale_array
)

# 3. Register the fake implementation for Dynamo tracing
@scale_op.register_fake
def _scale_op_fake(x, factor):
    return torch.empty_like(x)

# 4. Execute on TPU using the PyTorch-XLA runtime
if __name__ == "__main__":
    # Ensure execution occurs on the TPU device
    device = xm.xla_device()
    x_tensor = torch.ones(1024, device=device)
    
    # Run the custom operator
    out_tensor = scale_op(x_tensor, 5.0)
    print("TPU Output Tensor:", out_tensor) # Expected: Tensor filled with 5.0
```

---

## Pillar 3: Out-of-Tree (OOT) Platform Plugin Architecture

### 1. Concept
An out-of-tree platform plugin is a modular design pattern that allows external hardware runtimes (like TPUs or custom NPUs) to register themselves and override execution components in vLLM without modifying vLLM's core codebase.

### 2. Walkthrough
*   **Upstream Interface:** Upstream `vllm` defines a `Platform` abstract base class containing hooks like `check_and_update_config()`, `get_worker_distributed_backend()`, and `get_attn_backend_cls()`.
*   **Plugin Registration:** We register `vllm_torchtpu` under the `vllm.platform_plugins` entry point namespace in [pyproject.toml](../pyproject.toml#L299):
    ```toml
    [project.entry-points."vllm.platform_plugins"]
    torchtpu = "vllm_torchtpu.platforms:register_tpu_platform"
    ```
*   **Discovery:** When `vllm serve` or the `LLM` engine starts, it scans Python's metadata entry points.
*   **Instantiation:** It detects `torchtpu`, loads the [TpuPlatform](../src/vllm_torchtpu/platforms/tpu_platform.py) class, and instantiates it.
*   **Delegation:** vLLM queries `TpuPlatform` to determine the worker class (`TPUWorker`), the compiler adapter (`TpuCompilerAdaptor`), and the custom attention backend. The engine replaces its default CUDA execution paths with these TPU-specific overrides, allowing us to run on TPU with zero modifications to the core `vllm` repository.

### 3. Pseudocode
```python
# 1. vLLM Core Platform Discovery Loop
def discover_execution_platform():
    from importlib.metadata import entry_points
    
    # Scan for OOT platform plugins
    plugins = entry_points(group="vllm.platform_plugins")
    for plugin in plugins:
        register_fn = plugin.load()
        platform_cls = register_fn()
        if platform_cls.is_active():
            print(f"Discovered active platform: {platform_cls.device_name}")
            return platform_cls()
            
    return DefaultGPUMetadataPlatform()

# 2. vLLM Engine Initialization
platform = discover_execution_platform()
# Reconfigure parallel configurations dynamically based on the active platform
vllm_config.parallel_config.worker_cls = platform.get_worker_class()
```

### 4. Real Code (Minimal Simulation)
```python
# Simulation of the plugin registration and discovery interface
class AbstractPlatform:
    device_name = "generic"
    def check_and_update_config(self, config):
        pass

# Our TPU Platform Plugin implementation
class TpuPlatformPlugin(AbstractPlatform):
    device_name = "tpu"
    def check_and_update_config(self, config):
        config["device"] = "tpu"
        config["worker_cls"] = "vllm_torchtpu.worker.tpu_worker.TPUWorker"
        print("TpuPlatform: Updated configuration to TPU-specific classes.")

# Simulation of the entrypoint registry
entry_points_registry = {
    "vllm.platform_plugins": {
        "torchtpu": lambda: TpuPlatformPlugin
    }
}

# Simulation of vLLM Engine Discovery
def initialize_vllm_engine(config):
    print("Initializing vLLM Engine...")
    
    # Scan registry
    plugins = entry_points_registry.get("vllm.platform_plugins", {})
    active_platform = None
    for name, load_fn in plugins.items():
        platform_cls = load_fn()
        # Simulate check if hardware is present
        if platform_cls.device_name == "tpu":
            active_platform = platform_cls()
            break
            
    if active_platform:
        active_platform.check_and_update_config(config)
    else:
        print("Using default GPU/CPU execution path.")
        config["device"] = "cuda"

# Run the simulation
engine_config = {"device": "cuda", "worker_cls": "gpu_worker"}
initialize_vllm_engine(engine_config)
print("Final Engine Config:", engine_config)
```
