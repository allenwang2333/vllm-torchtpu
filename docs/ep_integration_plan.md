# Integration Plan: Enabling Expert Parallelism to Resolve TPU VMEM OOM

This document outlines the plan to resolve the JAX compilation `CompileTimeScopedVmemOom` error when running the full `DeepSeek-V4-Flash` model, by enabling Expert Parallelism (EP).

---

## 1. The Problem: VMEM Exhaustion

During compilation of the MXFP4 MoE GMM kernel, the JAX compiler attempts to allocate the sharded weight tensor on the stack in VMEM (Vector Memory):
*   **Physical Limit:** 32MB per TPU core.
*   **Required Allocation:** 40.01MB for `%copy = s8[256, 512, 8, 512]`.
*   **Root Cause:** With `tensor_parallel_size = 8` and `expert_parallelism = 1` (default), all 256 experts are replicated on every TPU chip. The GMM kernel materializes the weights for all 256 experts in VMEM at once, exceeding the physical limit.

---

## 2. Proposed Solution: Expert Parallelism (EP = 8)

We will shard the 256 experts across the 8 TPU chips, so each chip only manages **32 experts**.

### Sharding Math & VMEM Footprint:
*   **Local Experts per Chip:** $256 / 8 = 32$.
*   **New VMEM Allocation Shape:** `s8[32, 512, 8, 512]`.
*   **New VMEM Allocation Size:** $32 \times 512 \times 8 \times 512 \text{ bytes} \approx 5.00\text{MB}$.
*   **Safety Margin:** 5.00MB is well below the 32MB physical limit (84% reduction in VMEM footprint).

---

## 3. Options & Trade-offs

| Option | Sharding Config | VMEM Footprint | Pros | Cons |
| :--- | :--- | :--- | :--- | :--- |
| **Option 1 (Recommended)** | `TP=1, EP=8` | **5.00MB** | - Maximum memory relief for MoE layers.<br>- Fits easily within 32MB limit. | - Attention layers are replicated (unsharded). |
| **Option 2** | `TP=2, EP=4` | **10.00MB** | - Shards attention layers 2-way.<br>- Balances attention and MoE memory. | - Slightly higher VMEM footprint (still safe). |
| **Option 3** | `TP=4, EP=2` | **20.00MB** | - Shards attention layers 4-way. | - High VMEM footprint (closer to 32MB limit). |

We will implement a dynamic sharding configuration that allows us to easily test both `TP=1, EP=8` and `TP=2, EP=4`.

---

## 4. Detailed Implementation Steps

### Step 1: Modify `sharding.py`
We must update [sharding.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/layers/common/sharding.py) to dynamically scale `tensor_parallelism` based on `expert_parallelism`, so that the product of the sharding dimensions matches the allocated workers (`tensor_parallel_size`).

```python
# src/vllm_torchtpu/layers/common/sharding.py
# In ShardingConfigManager.from_vllm_config:

expert_parallelism = sharding_strategy.get("expert_parallelism", 1)
# Scale down tensor_parallelism so that TP * EP = tensor_parallel_size
tensor_parallelism = parallel_config.tensor_parallel_size // expert_parallelism
```

### Step 2: Configure `run_dsv4_inference.py`
Update the inference script to pass the `expert_parallelism` parameter via `additional_config`:

```python
# examples/run_dsv4_inference.py
llm = LLM(
    model="deepseek-ai/DeepSeek-V4-Flash",
    tensor_parallel_size=8,
    trust_remote_code=True,
    max_model_len=4096,
    kv_cache_dtype="fp8",
    gpu_memory_utilization=0.5,
    additional_config={
        "sharding": {
            "sharding_strategy": {
                "expert_parallelism": 8,  # Enables EP=8, TP=1
            }
        }
    }
)
```

### Step 3: Verify weight loading and compilation
1.  Sync the changes to the VM.
2.  Execute the inference script.
3.  Verify that:
    *   `w13_weight` is loaded with shape `[32, 512, 2048]` on each worker.
    *   The JAX GMM kernel compiles successfully without VMEM OOM.
    *   Tokens are generated.
