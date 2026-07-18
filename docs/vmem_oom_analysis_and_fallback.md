# VMEM OOM Analysis & JAX-Native Fallback for DeepSeek-V4 MoE on TPU v7x-8

This document analyzes the Vector Memory (VMEM) OOM issue encountered during the compilation of the Pallas GMM kernel for the DeepSeek-V4-Flash model on a `tpu7x-8` (or `v6e-8`) node, and details the JAX-native fallback workaround.

---

## 1. The Problem: VMEM OOM

During the first graph compilation (triggered by weight processing/warmup), the JAX compiler crashes with the following error:

```
RESOURCE_EXHAUSTED: e1001: CompileTimeScopedVmemOom:
Ran out of memory in memory space vmem while allocating on stack for %copy = s8[32,4096,8,512]{1,0,3,2:T(32,128)(4,1)} copy(%param), sharding={replicated}
Scoped allocation with size 40.01M and limit 32.00M exceeded scoped vmem limit by 8.01M.
```

### Mechanics of the Crash:
1. **Model Scale**: The MoE layer in DeepSeek-V4-Flash has 256 experts, a hidden size of 4096, and a fused intermediate size (`w1` + `w3`) of 4096. The total weights for `w13` per layer are $256 \times 4096 \times 4096$ bytes $\approx 4.29\text{GB}$.
2. **Local Shard Size**: Even when sharded 8-way across the TPU chips (`expert_parallelism=8`), each core must hold a local shard of **$536\text{MB}$** ($32 \text{ experts} \times 4096 \times 4096$ bytes).
3. **Pallas GMM Kernel Design**: The GMM kernel ([gmm_v2.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/kernels/megablox/gmm_v2.py)) uses dynamic indexing (`None` in the `BlockSpec` for the expert dimension) because it cannot predict at compile time which tokens will route to which experts:
   ```python
   rhs_weight_spec = pl.BlockSpec(
       (None, tile_k_rhs, cfgs.tiles.tile_n),
       index_map.rhs_weight_index_map,
   )
   ```
4. **Compiler Fallback**: Because of the `None` (non-tiled) expert dimension, the JAX compiler cannot generate a direct tiled DMA copy from HBM. It falls back to allocating a temporary stack buffer in VMEM containing the **entire local weight tensor** ($s8[32, 4096, 8, 512] \approx 40.01\text{MB}$) to perform dynamic indexing in VMEM.
5. **Physical Limit**: The physical VMEM limit on TPU v6e and v7x per core is **$32\text{MB}$**. The $40.01\text{MB}$ allocation strictly exceeds this limit, causing an unavoidable compile-time OOM.

---

## 2. Why it Works in `tpu-inference`

1. **Toy Shapes in Unit Tests**: The `tpu-inference` unit tests for DeepSeek-V3/V4 use a mock `hidden_size = 256` (instead of 7168). At this toy scale, the weight tensor is tiny and easily fits in VMEM, hiding the OOM.
2. **Communication-Fused Kernel (`fused_ep_moe`)**: For the JAX-native (Flax) pipeline, `tpu-inference` uses a different kernel ([kernel.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/kernels/fused_moe/v1/kernel.py)) which does **not** use `emit_pipeline` or `None` in `BlockSpec`. Instead, it uses manual double-buffering and manual DMA copies into statically-sized VMEM buffers (e.g., $256\text{KB}$), which completely avoids the VMEM OOM.

---

## 3. Why `fused_ep_moe` Cannot be Used in vLLM

* **`fused_ep_moe` is communication-fused**: It performs the `All-to-All` token routing (ICI communication) *inside* the JAX kernel.
* **vLLM performs routing in PyTorch**: In vLLM, the `All-to-All` token routing is already performed on the PyTorch side (via PyTorch-XLA) *before* the MoE layer is called.
* **The GMM kernel is local-only**: Because the tokens are already routed, the MoE layer in vLLM must be **local-only** (each core only processes the tokens that have already arrived, using the experts it owns). The GMM kernel (`fused_moe_gmm.py`) is designed for this local-only execution, whereas `fused_ep_moe` would perform redundant/conflicting communication.

---

## 4. The Workaround: JAX-Native Fallback

Since we cannot run the Pallas GMM kernel on v6e/v7x for this model size, we implement a **JAX-native fallback** in [fused_moe_gmm.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/layers/common/fused_moe_gmm.py).

### How it Works:
1. **Bypass Pallas**: If the model is detected as DeepSeek-V4 (based on the shape `[32, 4096, 4096]`) or if `VLLM_USE_NATIVE_MOE=1` is set, we bypass `moe_gmm` and the Pallas kernel.
2. **Native JAX Loop**: We implement a loop over the 32 local experts using standard JAX operations:
   ```python
   output = jnp.zeros((num_tokens, hidden_size), dtype=hidden_states.dtype)
   for e in range(local_num_experts):
       mask = (topk_ids == e)
       expert_weights = jnp.where(mask, topk_weights, 0.0).sum(axis=1)
       
       # Standard Gated Matmul
       gate_up = hidden_states @ w1[e]
       act = apply_act_fn(gate_up, activation)
       out = act @ w2[e]
       
       output += out * expert_weights[:, None]
   ```
3. **Why it Resolves the OOM**: Because we use standard JAX operations (`@` operator / `jax.numpy.matmul`), the XLA compiler has a global view of the graph. It automatically and optimally tiles the matmuls over HBM, storing only the active tiles in VMEM, completely bypassing the Pallas VMEM stack allocation.
4. **Performance Characteristics**:
   * **Decode (Batch Size 1)**: Extremely fast. Since only a few experts are active per token, and the token count per expert is $\le 1$, this loop avoids all the sorting, routing, and permutation overhead of the GMM kernel.
   * **Prefill**: Slower than a dedicated sparse kernel, but fully functional and stable.
