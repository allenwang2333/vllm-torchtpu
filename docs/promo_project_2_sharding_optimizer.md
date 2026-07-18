# Design Doc: Automated Sharding Strategy Optimizer for torchtpu-vllm

**Author:** [User's Name]  
**Status:** DRAFT  
**Target Capability:** L5 Promo Project (High Performance Impact)

---

## 1. Objective
Design and implement an automated sharding strategy optimizer within `torchtpu-vllm`. The optimizer will dynamically determine the most performant parallel execution configuration (Tensor, Expert, Data, and Attention-Data Parallelism) for any given model and TPU topology at startup.

---

## 2. Background & Problem Statement
Currently, configuring a model for multi-chip execution requires manually specifying parallel parameters in `additional_config` (e.g., `expert_parallelism`, `enable_dp_attention`). Finding the optimal configuration is highly complex. The current implementation suffers from two major limitations:

### A. Brittle Heuristics in Sharding Calculations
Our current sharding configuration relies on rigid, hardcoded division rules that do not adapt to different model architectures. For example, in `src/vllm_torchtpu/layers/common/sharding.py`, we calculate the attention data parallelism (`attn_dp`) using a simple division that can collapse `tensor_parallelism` to zero if the number of KV heads is small:

```python
# Current brittle implementation in sharding.py
class LegacyShardingManager:
    def __init__(self, tensor_parallel_size, num_kv_heads, use_mla, cache_dtype):
        # Hardcoded check for MLA
        num_kv_heads = 1 if use_mla else num_kv_heads
        packing = 4 if cache_dtype == "fp8" else 2
        
        # Division that collapses to 0 if num_kv_heads * 2 / packing < 1
        num_kv_heads_per_device = (num_kv_heads * 2) / packing # e.g. (1 * 2) / 4 = 0.5
        
        # This division yields 1 // 0.5 = 2.0 -> converted to int is 2
        attn_dp = int(tensor_parallel_size // num_kv_heads_per_device)
        
        # Crucial bug: 1 // 2 = 0. Tensor parallelism collapses to 0!
        self.tensor_parallelism = tensor_parallel_size // attn_dp
        self.attention_data_parallelism = attn_dp
```
This hardcoded math works for standard models (like Llama with MHA) but immediately crashes with a `ZeroDivisionError` or invalid JAX mesh shape when running models with MLA (like DeepSeek-V4) unless manually overridden.

### B. Memory-vs-Compute Contention (Trial-and-Error)
Determining whether to prioritize Tensor Parallelism (TP) or Expert Parallelism (EP) is currently done via manual trial-and-error. If an engineer makes a suboptimal choice, it leads to immediate Out-Of-Memory (OOM) errors or severe execution bubbles. For example, consider a manual configuration that replicates the KV cache unnecessarily:

```python
# Inefficient manual configuration for DeepSeek-V4 (TP=8, EP=1)
# Since DeepSeek-V4 has only 1 KV head (MLA), setting TP=8 replicates 
# the same KV head 8 times across the devices, wasting HBM:
class SuboptimalSharding:
    def __init__(self):
        self.tensor_parallelism = 8
        self.expert_parallelism = 1
        self.attention_data_parallelism = 1

    def calculate_kv_cache_memory(self, num_blocks, block_size, head_dim):
        # The KV cache is replicated 8 times because TP=8 but num_kv_heads=1.
        # This wastes 7/8ths of the allocated HBM:
        total_elements = num_blocks * block_size * self.tensor_parallelism * head_dim
        return total_elements * 2 # 2 bytes for BF16
```
An automated optimizer must dynamically balance this trade-off by modeling the memory footprint of the KV cache and the communication overhead of the MoE layer.

---

## 3. Proposed Design

We introduce a dedicated optimization module that replaces manual heuristics with a mathematical cost solver.

### A. Target File Structure
The sharding optimizer is isolated into a new module:
```
src/vllm_torchtpu/layers/common/
├── sharding.py          # Updated: References the optimizer during mesh creation
└── sharding_optimizer/
    ├── __init__.py      # Exports ShardingOptimizer
    ├── optimizer.py     # Contains the cost solver and search loop
    └── profiler.py      # Queries JAX devices to build the physical topology map
```

### B. Execution Flow Sequence
At model startup, the sharding strategy is resolved through the following sequence:

1.  **Topology Discovery:** The `TPUTopologyProfiler` queries JAX:
    ```python
    import jax
    # Query the physical devices
    devices = jax.devices()
    num_chips = len(devices)
    # Identify the interconnect topology (e.g. 2x2x1x2 mesh)
    bounds = jax.local_devices()[0].client.get_default_device_assignment(num_chips)
    ```
2.  **Strategy Space Generation:** The `ShardingOptimizer` generates all mathematically valid factorizations of the physical chip count:
    $$N = d_{\text{tp}} \times d_{\text{ep}} \times d_{\text{dp}} \times d_{\text{attn\_dp}}$$
3.  **Cost Evaluation:** For each candidate factorization, the optimizer calculates the estimated latency:
    ```python
    cost = estimate_attention_all_reduce(tp) + estimate_moe_all_to_all(ep) + estimate_compute_padding(tp)
    ```
4.  **Constraint Filtering:** The optimizer filters out any strategy that exceeds the per-device HBM limit:
    ```python
    if estimated_kv_cache_memory(tp, attn_dp) > available_hbm:
        continue # Discard this strategy
    ```
5.  **Mesh Allocation:** The optimal strategy is returned to `ShardingConfigManager`, which calls `build_mesh` and initializes the JAX device placement.

---

## 4. Optimization Solver Implementation

Here is the complete implementation of the cost solver:

```python
import numpy as np
from typing import List, Tuple

class TPUTopologyProfiler:
    def __init__(self):
        self.num_chips = len(jax.devices())
        # Mock bandwidth values based on TPU generation
        # v7x ICI bandwidth is approximately 450 GB/s
        self.ici_bandwidth_bytes_per_sec = 450.0 * 1e9 / 8

class ShardingOptimizer:
    def __init__(self, profiler: TPUTopologyProfiler):
        self.num_chips = profiler.num_chips
        self.bandwidth = profiler.ici_bandwidth_bytes_per_sec
        self.latency = 2.0 * 1e-6 # 2 microseconds flat latency
        self.w_tile = 128

    def get_valid_strategies(self) -> List[Tuple[int, int, int, int]]:
        strategies = []
        for tp in [1, 2, 4, 8]:
            for ep in [1, 2, 4, 8]:
                for dp in [1, 2, 4, 8]:
                    for attn_dp in [1, 2, 4, 8]:
                        if tp * ep * dp * attn_dp == self.num_chips:
                            strategies.append((tp, ep, dp, attn_dp))
        return strategies

    def evaluate_strategy(
        self,
        strategy: Tuple[int, int, int, int],
        hidden_size: int,
        num_experts: int,
        batch_size: int,
        seq_len: int,
    ) -> float:
        tp, ep, dp, attn_dp = strategy
        num_tokens = batch_size * seq_len
        
        # 1. Attention All-Reduce Cost (2 bytes per BF16 element)
        activation_size = num_tokens * hidden_size * 2
        t_ar = self.latency + (2 * (tp - 1) * activation_size) / (tp * self.bandwidth)
        
        # 2. MoE All-to-All Cost
        tokens_per_expert = num_tokens / num_experts
        t_a2a = self.latency + ((ep - 1) * tokens_per_expert * hidden_size * 2) / (ep * self.bandwidth)
        
        # 3. Compute Padding Cost
        local_hidden = hidden_size / tp
        padding_factor = (np.ceil(local_hidden / self.w_tile) * self.w_tile) / local_hidden
        t_compute = (num_tokens * hidden_size * hidden_size * 2) * padding_factor / 1e12
        
        return t_ar + t_a2a + t_compute

    def find_optimal_strategy(
        self,
        hidden_size: int,
        num_experts: int,
        batch_size: int,
        seq_len: int,
        available_hbm_bytes: float,
    ) -> Tuple[int, int, int, int]:
        strategies = self.get_valid_strategies()
        best_strategy = None
        min_cost = float("inf")
        
        for strategy in strategies:
            tp, ep, dp, attn_dp = strategy
            
            # Memory Constraint Check: Ensure KV cache does not overflow HBM
            # Replicating KV heads occurs if tp > 1 and attn_dp is not utilized
            kv_memory = (batch_size * seq_len * 128 * 2) * (tp / attn_dp)
            if kv_memory > available_hbm_bytes:
                continue
                
            cost = self.evaluate_strategy(
                strategy, hidden_size, num_experts, batch_size, seq_len
            )
            if cost < min_cost:
                min_cost = cost
                best_strategy = strategy
                
        if best_strategy is None:
            raise RuntimeError("No valid sharding strategy satisfied the HBM constraints.")
            
        return best_strategy
```
