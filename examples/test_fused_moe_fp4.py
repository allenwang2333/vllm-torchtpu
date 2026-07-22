import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
import numpy as np
import torch

from vllm_torchtpu.kernels.megablox.gmm_v2 import apply_act_fn
from vllm_torchtpu.layers.common.fused_moe_gmm import fused_moe_func
from vllm_torchtpu.layers.common.quantization import pack_fp4_indices

# Reference online unpack logic inside the GMM Pallas kernel (translated to NumPy/PyTorch CPU format)
def unpack_gmm_weights(w_packed: np.ndarray, axis: int) -> np.ndarray:
    # Extract nibbles
    low_nibble = w_packed & 0x0F
    high_nibble = (w_packed >> 4) & 0x0F
    
    # OCP float4 lookup table mapping index (0-15) to float
    LUT = np.array([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0,
        -3.0, -4.0, -6.0
    ], dtype=np.float32)
    
    low_val = LUT[low_nibble]
    high_val = LUT[high_nibble]
    
    # Interleave low and high along the specified axis
    stacked = np.stack([low_val, high_val], axis=axis + 1)
    
    # Reshape to merge the stacked axis back into the target axis
    new_shape = list(w_packed.shape)
    new_shape[axis] = w_packed.shape[axis] * 2
    return stacked.reshape(new_shape)

def _reference_fused_moe_fp4(
    hidden_states, w1_packed, w2_packed, 
    w1_scale, w2_scale, w1_bias, w2_bias, 
    topk_weights, topk_ids, activation, block_size
):
    num_tokens, hidden_size = hidden_states.shape
    
    # Dequantize weights on CPU
    w1_unpacked = unpack_gmm_weights(np.asarray(w1_packed), axis=1) # [num_experts, K, N1]
    w2_unpacked = unpack_gmm_weights(np.asarray(w2_packed), axis=1) # [num_experts, intermediate, K]
    
    # Squeeze the 1-dim axis from the 4D scale tensors to repeat them
    w1_scale_np = np.squeeze(np.asarray(w1_scale), axis=2) # [num_experts, K // block_size, N1]
    w2_scale_np = np.squeeze(np.asarray(w2_scale), axis=2) # [num_experts, intermediate // block_size, K]
    
    # Repeat scales along the blocked contracting dimension (axis 1 for both)
    w1_scale_expanded = np.repeat(w1_scale_np, block_size, axis=1) # [num_experts, K, N1]
    w2_scale_expanded = np.repeat(w2_scale_np, block_size, axis=1) # [num_experts, intermediate, K]
    
    w1_dequant = w1_unpacked * w1_scale_expanded
    w2_dequant = w2_unpacked * w2_scale_expanded
    
    padded_hidden_size = w1_dequant.shape[1]
    hidden_states = jnp.pad(hidden_states, ((0, 0), (0, padded_hidden_size - hidden_size)))
    out = jnp.zeros((num_tokens, padded_hidden_size), dtype=jnp.float32)
    
    for token_id in range(num_tokens):
        token_out = jnp.zeros((padded_hidden_size, ), dtype=jnp.float32)
        for expert_slot in range(topk_ids.shape[1]):
            expert_id = int(topk_ids[token_id, expert_slot])
            if expert_id < 0:
                continue
                
            gate_up = jnp.matmul(hidden_states[token_id].astype(jnp.float32), w1_dequant[expert_id])
            if w1_bias is not None:
                gate_up = gate_up + w1_bias[expert_id, 0].astype(jnp.float32)
                
            activated = apply_act_fn(gate_up[None], activation)[0]
            proj = jnp.matmul(activated.astype(jnp.float32), w2_dequant[expert_id])
            if w2_bias is not None:
                proj = proj + w2_bias[expert_id, 0].astype(jnp.float32)
                
            token_out = token_out + topk_weights[token_id, expert_slot].astype(jnp.float32) * proj
        out = out.at[token_id].set(token_out)
        
    return out[:, :hidden_size].astype(hidden_states.dtype)

def test_fused_moe_fp4_parity():
    # Shapes matching Qwen3 MoE shapes
    num_tokens = 4
    hidden_size = 2048
    padded_hidden_size = 2048
    intermediate_size = 768
    num_experts = 16
    topk = 8
    block_size = 128
    
    # 1. Initialize random inputs
    key = jax.random.key(42)
    hk, w1k, w2k, s1k, s2k, b1k, b2k = jax.random.split(key, 7)
    
    # Scale down inputs to prevent huge values and accumulation range rounding errors
    hidden_states = (jax.random.normal(hk, (num_tokens, hidden_size), dtype=jnp.float32) / 10).astype(jnp.bfloat16)
    
    # 2. Build mock FP4 expert indices (0-15 indices to pack)
    w1_indices = torch.randint(0, 16, (num_experts, padded_hidden_size, intermediate_size * 2), dtype=torch.uint8)
    w2_indices = torch.randint(0, 16, (num_experts, intermediate_size, padded_hidden_size), dtype=torch.uint8)
    
    # Pack them along K axis (axis 1 for w1, axis 2 for w2)
    w1_packed = pack_fp4_indices(w1_indices, axis=1) # shape [16, 1024, 1536]
    w2_packed = pack_fp4_indices(w2_indices, axis=1) # shape [16, 384, 2048]
    
    w1_packed_jax = jnp.array(w1_packed.numpy())
    w2_packed_jax = jnp.array(w2_packed.numpy())
    
    # Scale down weights scales
    w1_scale = (jax.random.normal(s1k, (num_experts, padded_hidden_size // block_size, 1, intermediate_size * 2), dtype=jnp.float32) / 10).astype(jnp.bfloat16)
    w2_scale = (jax.random.normal(s2k, (num_experts, intermediate_size // block_size, 1, padded_hidden_size), dtype=jnp.float32) / 10).astype(jnp.bfloat16)
    
    w1_bias = (jax.random.normal(b1k, (num_experts, 1, intermediate_size * 2), dtype=jnp.float32) / 10).astype(jnp.bfloat16)
    w2_bias = (jax.random.normal(b2k, (num_experts, 1, padded_hidden_size), dtype=jnp.float32) / 10).astype(jnp.bfloat16)
    
    topk_weights = jnp.array([
        [0.30, 0.18, 0.14, 0.12, 0.10, 0.07, 0.05, 0.04],
        [0.22, 0.18, 0.16, 0.14, 0.11, 0.08, 0.06, 0.05],
        [0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04],
        [0.28, 0.19, 0.15, 0.11, 0.09, 0.07, 0.06, 0.05],
    ], dtype=jnp.float32)
    
    topk_ids = jnp.array([
        [0, 2, 4, 6, 8, 10, 12, 14],
        [1, 3, 5, 7, 9, 11, 13, 15],
        [15, 13, 11, 9, 7, 5, 3, 1],
        [14, 12, 10, 8, 6, 4, 2, 0],
    ], dtype=jnp.int32)
    
    print("Running TPU JAX fused_moe_func kernel...")
    actual = fused_moe_func(
        hidden_states=hidden_states,
        w1=w1_packed_jax,
        w2=w2_packed_jax,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        topk=topk,
        activation="silu",
        use_sparse_core=False # TPU VM local run plain JAX path
    )
    
    print("Running CPU reference dequantization + eager GMM...")
    expected = _reference_fused_moe_fp4(
        hidden_states, w1_packed_jax, w2_packed_jax,
        w1_scale, w2_scale, w1_bias, w2_bias,
        topk_weights, topk_ids, "silu", block_size
    )
    
    actual_np = np.asarray(actual, dtype=np.float32)
    expected_np = np.asarray(expected, dtype=np.float32)
    
    diff = np.abs(actual_np - expected_np)
    max_diff = float(np.max(diff))
    mean_abs_err = float(np.mean(diff))
    
    # Calculate relative diff where expected is non-zero
    non_zero_mask = np.abs(expected_np) > 1e-4
    rel_diff = np.where(non_zero_mask, diff / np.abs(expected_np), 0.0)
    max_rel_diff = float(np.max(rel_diff))
    
    print(f"FP4 Fused MoE Matmul Max Absolute Difference: {max_diff:.6f}")
    print(f"FP4 Fused MoE Matmul Mean Absolute Error:     {mean_abs_err:.6f}")
    print(f"FP4 Fused MoE Matmul Max Relative Difference: {max_rel_diff:.6f}")
    
    # Assert correctness using typical BF16 tolerance limits (0.15 atol, 0.1 rtol)
    np.testing.assert_allclose(actual_np, expected_np, atol=0.15, rtol=1e-1)
    print("SUCCESS: Parity verified!")

if __name__ == "__main__":
    test_fused_moe_fp4_parity()
