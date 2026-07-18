import os
import torch
import numpy as np
import jax
import jax.numpy as jnp
import functools
from vllm_torchtpu.layers.common.fused_moe_gmm import fused_moe_func

# Initialize PyTorch TPU device
device = torch.device("tpu")
print(f"Using device: {device}")

# JAX Mesh
from jax.sharding import Mesh
devices = jax.devices()
mesh = Mesh(np.array([devices[0]]).reshape(1, 1), ('attn_data', 'attn_head'))

# Context
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import set_vllm_model_wrapper_context
from torch_tpu._internal import pallas

# Dimensions matching DeepSeek-V4 expert layers (scaled down experts count for unit test)
num_tokens = 16
hidden_size = 4096
local_experts = 8
intermediate_dim = 2048
topk = 2

# Generate random weights and inputs
hidden_states_cpu = np.random.randn(num_tokens, hidden_size).astype(np.float32)
w1_cpu = np.random.randn(local_experts, intermediate_dim * 2, hidden_size).astype(np.float32) * 0.02
w2_cpu = np.random.randn(local_experts, hidden_size, intermediate_dim).astype(np.float32) * 0.02

# For FP8 scale weights (or None for unquantized)
w1_scale_cpu = np.random.rand(local_experts, 1, 1).astype(np.float32) + 0.5
w2_scale_cpu = np.random.rand(local_experts, 1, 1).astype(np.float32) + 0.5

# Gating choices
topk_weights_cpu = np.random.rand(num_tokens, topk).astype(np.float32)
topk_weights_cpu /= topk_weights_cpu.sum(axis=-1, keepdims=True)
topk_ids_cpu = np.random.randint(0, local_experts, (num_tokens, topk)).astype(np.int32)

# Transfer to PyTorch TPU device
hidden_states_torch = torch.from_numpy(hidden_states_cpu).to(torch.bfloat16).to(device)
w1_torch = torch.from_numpy(w1_cpu).to(torch.bfloat16).to(device)
w2_torch = torch.from_numpy(w2_cpu).to(torch.bfloat16).to(device)
w1_scale_torch = torch.from_numpy(w1_scale_cpu).to(torch.bfloat16).to(device)
w2_scale_torch = torch.from_numpy(w2_scale_cpu).to(torch.bfloat16).to(device)
topk_weights_torch = torch.from_numpy(topk_weights_cpu).to(torch.bfloat16).to(device)
topk_ids_torch = torch.from_numpy(topk_ids_cpu).to(device)

# Wrap the JAX fused_moe_func in a PyTorch Custom Op
with set_vllm_model_wrapper_context(mesh=mesh):
    wrapped_fn = functools.partial(
        fused_moe_func,
        topk=topk,
        activation="silu",
        use_ep=False,
        use_sparse_core=False, # run on TPU Tensor Cores
    )
    moe_op = pallas.jax_op("pallas::deepseek_v4_fused_moe_test", wrapped_fn)
    
    def _fake_moe(hs, w1, w2, *args):
        return torch.empty((hs.shape[0], w2.shape[1]), dtype=hs.dtype, device=hs.device)
    moe_op.register_fake(_fake_moe)

print("Running PyTorch Custom Op on TPU (Megablox GMM)...")
# Force GMM mode
os.environ["VLLM_FORCE_GMM"] = "1"
os.environ["VLLM_USE_NATIVE_MOE"] = "0"
output_tpu = moe_op(
    hidden_states_torch,
    w1_torch,
    w2_torch,
    w1_scale_torch,
    w2_scale_torch,
    None, # w1_bias
    None, # w2_bias
    topk_weights_torch,
    topk_ids_torch,
)
output_tpu_cpu = output_tpu.cpu().to(torch.float32).numpy()

print("Running JAX Native Reference on CPU (Loops)...")
# Convert inputs to JAX arrays
hidden_states_jax = jnp.array(hidden_states_cpu).astype(jnp.bfloat16)
w1_jax = jnp.array(w1_cpu).astype(jnp.bfloat16)
w2_jax = jnp.array(w2_cpu).astype(jnp.bfloat16)
w1_scale_jax = jnp.array(w1_scale_cpu).astype(jnp.bfloat16)
w2_scale_jax = jnp.array(w2_scale_cpu).astype(jnp.bfloat16)
topk_weights_jax = jnp.array(topk_weights_cpu).astype(jnp.bfloat16)
topk_ids_jax = jnp.array(topk_ids_cpu)

# Force native loop mode
os.environ["VLLM_FORCE_GMM"] = "0"
os.environ["VLLM_USE_NATIVE_MOE"] = "1"
output_ref = fused_moe_func(
    hidden_states_jax,
    w1_jax,
    w2_jax,
    w1_scale_jax,
    w2_scale_jax,
    None,
    None,
    topk_weights_jax,
    topk_ids_jax,
    topk=topk,
    activation="silu",
    use_ep=False,
    use_sparse_core=False,
)
output_ref_cpu = np.array(output_ref).astype(np.float32)

# Compare parity
max_diff = np.max(np.abs(output_tpu_cpu - output_ref_cpu))
print(f"Max absolute difference in MoE output: {max_diff}")
assert max_diff < 1e-2, f"MoE parity check failed! Max diff: {max_diff}"
print("MoE PARITY CHECK PASSED!")
