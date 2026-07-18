# Technical Deep Dive: DeepSeek-V4 Integration Details & Challenges

Integrating DeepSeek-V4 into `torchtpu-vllm` required bridging PyTorch's local-shape tracing paradigm and JAX's global-shape sharding paradigm. This document details the specific challenges ("hoops") we encountered and how they were resolved.

---

## Hoop 1: The Local-Shape Invariant in PyTorch Dynamo

### The Challenge
PyTorch's compiler (Dynamo) traces the model during `torch.compile`. Because the physical execution is sharded across devices (e.g. TP=8), all tensors in the PyTorch graph representing sharded data (like attention queries) **must maintain their local shapes**.
In the original implementation, the query tensor was reshaped using the global head count:
```python
q = q.view(-1, 64, 128)  # 64 is the global head count
```
However, at runtime, the PyTorch tensor `q` on each device only contained its local shard of heads, meaning its actual shape was `[8192, 8, 128]` (for TP=8). Attempting to view a tensor of size `8192 * 8 * 128` as `[-1, 64, 128]` caused a PyTorch runtime shape mismatch error:
```
RuntimeError: shape '[-1, 64, 128]' is invalid for input of size 8388608
```

### The Solution
We modified the query reshaping in [attention_impl](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py#L921) to use the local head count:
```python
q = self.wq_b(qr).view(qr.shape[0], self.mla_attn.n_local_heads, self.mla_attn.head_dim)
```
This preserves the local shape in the PyTorch graph. When the JAX compiler lowers this custom op, it references the `input_partition_specs=(P(None, "model", None), ...)` passed to `pallas.jax_op` and automatically scales the head dimension back to the global `64` inside JAX.

---

## Hoop 2: JAX Export Stripping Sharding (The `pallas.py` Patch)

### The Challenge
This was the most complex bug. During compilation, `pallas.py` uses `jax.export.export` to compile the JAX attention function to an MLIR module. However, `jax.export` treats the export boundary as replicated (`P()`), stripping all sharding information from the output avals.
As a result:
1.  `pallas.py`'s `torch_placeholder` function (which determines the shape of the PyTorch tensor returned by the custom op) saw a replicated output.
2.  It returned a tensor with the **global shape** `[8192, 64, 512]` to the PyTorch FX graph.
3.  This global-shaped tensor was then passed to the subsequent sharded layer, `group_proj_op`.
4.  During JAX compilation of the `group_proj_op`, the JAX backend compiler saw a sharded input spec `P(None, "model", None)` and assumed the incoming PyTorch tensor had a local shape.
5.  It scaled the head dimension of the PyTorch tensor (`64`) by the TP size (`8`), resulting in a JAX global shape of `[8192, 512, 512]`.
6.  This caused a massive shape mismatch inside the group projection matmul, crashing with:
    ```
    ValueError: Input hidden dim 32768 must match weight hidden dim 4096.
    ```
    *(32768 comes from the scaled local shape: 512 heads * 64 head_dim = 32768)*

### The Solution
We patched [pallas.py](file:///home/patemotter_google_com/workspace/dsv4/torchtpu-vllm/.venv/lib/python3.12/site-packages/torch_tpu/_internal/pallas/pallas.py) in the VM's virtual environment to support `output_partition_specs` in `jax_op`. 
```python
# In pallas.py (patched)
def torch_placeholder(tensor, mesh=None, partition_spec=None):
  ...
  spec = partition_spec if partition_spec is not None else tensor.sharding.spec
  return torch.empty(get_local_shape(tensor.shape, mesh, spec), ...)
```
By passing `output_partition_specs=(P(None, "model", None), P())` to `attn_jax_op`, we force `torch_placeholder` to return the local shape `[8192, 8, 512]` to PyTorch, preserving the local-shape invariant.

---

## Hoop 3: Indexer Weights Rank-2 Mismatch

### The Challenge
The indexer op `indexer_jax_op` selects the top-k KV blocks. Its second input, `indexer_weights`, is a 2D tensor representing the routing weights for each token across the heads, with shape `[t, 64]` (global) or `[t, 8]` (local).
We initially annotated all sharded inputs of the indexer using the same 3D partition spec:
```python
input_partition_specs=(
    P(None, "model", None),  # q_rope (rank 3)
    P(None, "model", None),  # indexer_weights (rank 2) <-- WRONG
    ...
)
```
Because `indexer_weights` is a 2D tensor, JAX crashed during compilation because we tried to apply a rank-3 sharding spec to a rank-2 tensor:
```
RuntimeError: Sharding NamedSharding(...) is only valid for values of rank at least 3, but was applied to a value of rank 2.
```

### The Solution
We updated [_build_indexer_op](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py#L601) to use the correct rank-2 sharding spec for `indexer_weights`:
```python
input_partition_specs=(
    P(None, "model", None),  # q_rope
    P(None, "model"),        # indexer_weights (Corrected to rank 2)
    ...
)
```

---

## Hoop 4: MoE VMEM OOM on TPU v6e/v7x

### The Challenge
DeepSeek-V4 uses a Mixture-of-Experts (MoE) layer with 256 experts. The standard implementation uses a Pallas GMM (Grouped Matrix Multiplication) kernel. This kernel requires allocating large static buffers in the TPU's Vector Memory (VMEM). Because TPU v6e and v7x have smaller VMEM limits compared to larger TPU architectures, compiling the GMM kernel for 256 experts caused a VMEM OOM error.

### The Solution
We implemented a JAX-native MoE fallback in [fused_moe_gmm.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/layers/common/fused_moe_gmm.py).
Instead of a custom Pallas kernel, the fallback uses `jax.shard_map` and standard JAX matrix multiplications (`jnp.matmul`). JAX compiles these operations into highly optimized XLA code that automatically manages memory tiling and paging, avoiding the manual VMEM allocation limits of the Pallas kernel while maintaining equivalent performance.

---

## Hoop 5: MoE FP8 Weight Loading & Sharding Mechanics

### The Challenge (Scale Parameter Naming Gotcha)
During checkpoint loading in vLLM, FP8 MoE weights and scaling factors undergo block dequantization and requantization in `_process_fp8_moe_weights` ([fp8.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/layers/vllm/quantization/fp8.py)).
*   **The Gotcha:** In vLLM/TPU parameter naming conventions, the scale parameter attached to `FusedMoE` is named `w13_weight_scale_inv` and `w2_weight_scale_inv`. Despite the `_inv` suffix in the parameter name, the values stored in this tensor are the **multiplicative scales** ($s = \frac{\text{abs\_max}}{\text{dtype\_max}}$), NOT the inverse scales ($\frac{1}{s}$).
*   **Mechanical Verification:** In [gmm_v2.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/kernels/megablox/gmm_v2.py), the kernel explicitly computes `block_scale_inv = jnp.where(block_scale == 0, 0, 1 / block_scale)` and multiplies dequantized activations by `rhs_scale_slice`. Passing inverse scales into this pipeline results in double-inversion ($s^2$ scaling errors) or division-by-zero NaNs.

### The Solution (Sharding Spec Alignment)
In `tpu-inference` (reference repo), MoE weights are loaded as global tensors in HBM, and `tensor_parallel_gmm` explicitly passes sharded partition specs to `jax.shard_map`:
```python
w1_spec = P(None, None, ShardingAxisName.MLP_TENSOR)  # Sharded on contracting dim
w2_spec = P(None, ShardingAxisName.MLP_TENSOR, None)  # Sharded on input dim
```
In `torchtpu-vllm`, because weight loading slices `w13_weight` and `w2_weight` to local physical shards on each TP worker process during model initialization, passing sharded specs to `shard_map` would slice an already-sharded tensor a second time ($\frac{4096}{8} \rightarrow 512 \rightarrow 64$). To preserve the local-shape invariant across the PyTorch XLA bridge without double-sharding, `w1_spec` and `w2_spec` in `tensor_parallel_gmm` must align with the physical sharding state of the incoming PyTorch tensors.
