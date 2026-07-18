# Design Doc: Unified TPU Attention Framework for torchtpu-vllm

**Author:** [User's Name]  
**Status:** DRAFT  
**Target Capability:** L5 Promo Project (Complexity & Architecture)

---

## 1. Objective
Replace the current model-specific attention patching mechanism in `torchtpu-vllm` with a unified, model-agnostic `TpuAttention` layer. This framework will dynamically select and execute the optimal TPU-optimized attention kernel (MLA, SWA, or standard MHA) based on the model's configuration at startup.

---

## 2. Background & Problem Statement
Currently, integrating a new model architecture requires writing custom patching scripts (such as `deepseek_v4_patch.py` or `llama_patch.py`) to swap out the model's attention classes with TPU-specific subclasses. This approach has three major drawbacks:

### A. Code Duplication in Metadata Extraction
Every model-specific attention layer must manually extract the same metadata tensors from the runner's context and perform the same shape assertions. For example, compare the duplication between the DeepSeek-V4 attention patch and the Llama attention patch:

```python
# Duplicated block in vllm_torchtpu/models/vllm/deepseek_v4_patch.py
class VllmDeepseekV4MLAAttention(torch.nn.Module):
    def forward(self, query, key, value):
        context = get_forward_context()
        attn_metadata = context.attn_metadata.get("deepseek_v4", None)
        if attn_metadata is None:
            raise ValueError("Missing attention metadata")
        seq_lens = attn_metadata.seq_lens
        block_tables = attn_metadata.block_tables
        query_start_loc = attn_metadata.query_start_loc
        request_distribution = attn_metadata.request_distribution
        return self.attn_op(query, key, seq_lens, block_tables, query_start_loc, request_distribution)

# Duplicated block in vllm_torchtpu/models/vllm/llama_patch.py
class VllmLlamaAttention(torch.nn.Module):
    def forward(self, query, key, value):
        context = get_forward_context()
        attn_metadata = context.attn_metadata.get("llama", None)
        if attn_metadata is None:
            raise ValueError("Missing attention metadata")
        seq_lens = attn_metadata.seq_lens
        block_tables = attn_metadata.block_tables
        query_start_loc = attn_metadata.query_start_loc
        request_distribution = attn_metadata.request_distribution
        return self.attn_op(query, key, value, seq_lens, block_tables, query_start_loc, request_distribution)
```
Both classes duplicate the exact same 8 lines of context extraction and validation logic.

### B. Maintenance Overhead from Monkey-Patching
To integrate these classes, we currently use dynamic monkey-patching in the model loader. This is highly vulnerable to upstream changes. For example:

```python
# In vllm_torchtpu/models/vllm/deepseek_v4_patch.py
def patch_deepseek_v4():
    import vllm.models.deepseek_v4.amd.model as ds_model
    # Overwrite the class definition at runtime
    ds_model.DeepseekV4Attention = VllmDeepseekV4MLAAttention
```
If an upstream vLLM update modifies `DeepseekV4Attention` to accept a new argument in its `forward` method (for example, `def forward(self, query, key, value, hidden_states)`), our patched `forward` method (which only accepts `query, key, value`) will raise a `TypeError` at runtime. Because this is resolved dynamically via monkey-patching, these errors are not caught by static linters or type checkers during compilation.

### C. Siloed Optimizations (Lack of Generality)
When an optimization is implemented for one attention variant, it remains siloed in that specific file. For example, if we implement FP8 KV cache scaling factors in `VllmLlamaAttention`:

```python
# Optimization implemented only in Llama
class VllmLlamaAttention(torch.nn.Module):
    def apply_fp8_scaling(self, key, value, scale):
        return key * scale, value * scale
```
The `VllmDeepseekV4MLAAttention` class cannot benefit from this scaling logic because it does not share a common base class or interface with `VllmLlamaAttention`, requiring us to copy-paste the scaling logic and maintain it in two places.

---

## 3. Proposed Architecture

We propose a unified `TpuAttention` layer that acts as a factory and dispatcher. 

```
   [vLLM Model Loader]
           |
           v
     [TpuAttention]  (Factory)
           |
           +---> [TpuMlaAttention]  (DeepSeek-V4, etc.)
           |
           +---> [TpuSwaAttention]  (Mistral, etc.)
           |
           +---> [TpuMhaAttention]  (Llama, etc.)
```

### A. The Attention Metadata Provider
To eliminate duplicate metadata extraction code, we introduce a unified `TpuAttentionMetadata` class that extracts and validates tensors from the active `ForwardContext` on the TPU.

```python
import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class TpuAttentionMetadata:
    seq_lens: torch.Tensor
    block_tables: torch.Tensor
    query_start_loc: torch.Tensor
    request_distribution: torch.Tensor

    @classmethod
    def extract(cls, prefix: str) -> "TpuAttentionMetadata":
        # Get the global forward context from the runner
        from vllm_torchtpu.runner.tpu_runner import get_forward_context
        context = get_forward_context()
        attn_metadata = context.attn_metadata.get(prefix, None)
        if attn_metadata is None:
            raise ValueError(f"Attention metadata not found for cache prefix: {prefix}")
        
        return cls(
            seq_lens=attn_metadata.seq_lens,
            block_tables=attn_metadata.block_tables,
            query_start_loc=attn_metadata.query_start_loc,
            request_distribution=attn_metadata.request_distribution,
        )
```

### B. The TpuAttention Factory Class
The unified `TpuAttention` module replaces the attention layer in the model. It delegates execution to specific implementations based on the model configuration.

```python
from vllm.config import ModelConfig, CacheConfig

class TpuAttention(torch.nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        prefix: str,
    ):
        super().__init__()
        self.model_config = model_config
        self.cache_config = cache_config
        self.prefix = prefix
        
        # Determine the attention variant and instantiate the implementation
        if getattr(model_config, "use_mla", False):
            self.impl = TpuMlaAttention(model_config, cache_config, prefix)
        elif getattr(model_config, "sliding_window", None) is not None:
            self.impl = TpuSwaAttention(model_config, cache_config, prefix)
        else:
            self.impl = TpuMhaAttention(model_config, cache_config, prefix)
            
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        # Extract common metadata
        metadata = TpuAttentionMetadata.extract(self.prefix)
        # Dispatch to the specific implementation
        return self.impl(positions, hidden_states, query, key, value, metadata)
```

### C. Example Implementation: `TpuMlaAttention`
Here is how the DeepSeek-V4 MLA implementation is encapsulated under the new framework:

```python
class TpuMlaAttention(torch.nn.Module):
    def __init__(self, model_config: ModelConfig, cache_config: CacheConfig, prefix: str):
        super().__init__()
        self.prefix = prefix
        self.n_local_heads = model_config.hf_config.num_attention_heads // 8 # TP=8 example
        self.head_dim = 512
        # Initialize the Pallas op using the unified config
        from vllm_torchtpu.layers.vllm.custom_ops.deepseek_v4 import _build_attn_op
        self.attn_op = _build_attn_op(model_config, cache_config, prefix)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: TpuAttentionMetadata,
    ) -> torch.Tensor:
        # Perform local reshaping
        q = query.view(query.shape[0], self.n_local_heads, self.head_dim)
        
        # Execute the custom JAX op using the common metadata
        output, _ = self.attn_op(
            q,
            key,
            metadata.seq_lens,
            metadata.block_tables,
            metadata.query_start_loc,
            metadata.request_distribution,
        )
        return output
```

### D. Target File Structure
The framework is split into discrete files to maintain modularity:
```
src/vllm_torchtpu/layers/vllm/attention/
├── __init__.py          # Exports TpuAttention and apply_global_tpu_patches
├── factory.py           # Contains the TpuAttention dispatcher class
├── metadata.py          # Contains the TpuAttentionMetadata extractor
├── mha.py               # Contains TpuMhaAttention (Llama, etc.)
├── mla.py               # Contains TpuMlaAttention (DeepSeek-V4, etc.)
└── swa.py               # Contains TpuSwaAttention (Mistral, etc.)
```

### E. Execution Flow Sequence
When a model executes inference, the control flow proceeds as follows:
1.  **Model Loading:** `TPUModelRunner` calls `apply_global_tpu_patches()`. This replaces the model's native attention classes (e.g., `DeepseekV4Attention`) with our unified `TpuAttention` class.
2.  **Forward Pass Triggered:** During a forward pass, the model calls `TpuAttention.forward()`.
3.  **Metadata Extraction:** `TpuAttention` calls `TpuAttentionMetadata.extract()`, which retrieves the current token positions, sequence lengths, and block tables from the global `ForwardContext`.
4.  **Variant Dispatching:** `TpuAttention` checks its internal `self.impl` (configured at startup) and calls the forward method of the specific subclass (e.g., `TpuMlaAttention.forward()`), passing the extracted metadata.
5.  **Pallas Kernel Execution:** The subclass executes the JAX-native Pallas kernel (e.g., `attn_op`) using the metadata and returns the output tensor back to the PyTorch graph.

### F. Handling Shared Sub-Operations via Composition
In a unified framework, some operations are shared by a *subset* of attention variants, but not all. For example:
*   **RoPE (Rotary Position Embeddings):** Used by `TpuMhaAttention` and `TpuSwaAttention` (which require standard key/value rotation), but **not** by `TpuMlaAttention` (which uses a specialized decoupled RoPE path).
*   **FP8 Scaling:** Used only when `kv_cache_dtype="fp8"`.

To prevent the dispatcher (`TpuAttention`) from becoming a "God Object" containing logic that only some variants need, we use **Composition** with standalone helper components.

#### 1. The RoPE Helper Component
We define a reusable `TpuRoPEHelper` class in `src/vllm_torchtpu/layers/vllm/attention/rope.py`:

```python
import torch

class TpuRoPEHelper(torch.nn.Module):
    def __init__(self, head_size: int, rotary_dim: int, max_position: int = 8192):
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        # Precompute cos and sin caches
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
        t = torch.arange(max_position, dtype=torch.float32)
        freqss = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqss.cos())
        self.register_buffer("sin_cached", freqss.sin())

    def forward(self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extract cos and sin for the active token positions
        cos = self.cos_cached[positions].unsqueeze(1) # Shape: [batch, 1, rotary_dim]
        sin = self.sin_cached[positions].unsqueeze(1)
        
        # Apply rotation to query and key
        q_rot = (q[..., :self.rotary_dim] * cos) + (rotate_half(q[..., :self.rotary_dim]) * sin)
        k_rot = (k[..., :self.rotary_dim] * cos) + (rotate_half(k[..., :self.rotary_dim]) * sin)
        
        # Reconstruct full tensors
        q_out = torch.cat([q_rot, q[..., self.rotary_dim:]], dim=-1)
        k_out = torch.cat([k_rot, k[..., self.rotary_dim:]], dim=-1)
        return q_out, k_out

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)
```

#### 2. Composing the Helper in MHA (Llama)
The Llama implementation instantiates and calls the RoPE helper:

```python
class TpuMhaAttention(torch.nn.Module):
    def __init__(self, model_config: ModelConfig, cache_config: CacheConfig, prefix: str):
        super().__init__()
        # Compose the RoPE helper
        self.rope_helper = TpuRoPEHelper(
            head_size=model_config.hf_config.hidden_size // model_config.hf_config.num_attention_heads,
            rotary_dim=model_config.hf_config.hidden_size // model_config.hf_config.num_attention_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: TpuAttentionMetadata,
    ) -> torch.Tensor:
        # Apply RoPE via composition
        q_rotated, k_rotated = self.rope_helper(positions, query, key)
        
        # Continue with MHA execution using rotated tensors...
        return execution_with_mha_kernel(q_rotated, k_rotated, value, metadata)
```

#### 3. Excluding the Helper in MLA (DeepSeek-V4)
The DeepSeek-V4 MLA implementation does **not** instantiate the RoPE helper. Instead, it directly passes the unrotated query and key to the custom MLA kernel (which handles its own internal decoupled RoPE scaling):

```python
class TpuMlaAttention(torch.nn.Module):
    def __init__(self, model_config: ModelConfig, cache_config: CacheConfig, prefix: str):
        super().__init__()
        # No RoPE helper is instantiated here

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: TpuAttentionMetadata,
    ) -> torch.Tensor:
        # Directly execute the MLA JAX kernel (it handles its own decoupled RoPE)
        return self.attn_op(query, key, metadata)
```

By using composition, each attention variant owns its specific execution logic, while shared utilities (like RoPE or FP8 scaling) are isolated into reusable, single-responsibility helper classes.

---

## 5. Model-Agnostic Patching Integration

Instead of having separate patch files for each model, we implement a centralized patching registry in `src/vllm_torchtpu/models/vllm/patching_registry.py`.

```python
def apply_global_tpu_patches():
    # Patch Llama
    import vllm.models.llama as llama_model
    llama_model.LlamaAttention = TpuAttention

    # Patch DeepSeek-V4
    import vllm.models.deepseek_v4.amd.model as ds_v4_model
    ds_v4_model.DeepseekV4Attention = TpuAttention
    
    print("Successfully applied global TPU attention patches.")
```
This function is called once during `load_model` in `TPUModelRunner`, ensuring all models automatically use the unified attention framework.
