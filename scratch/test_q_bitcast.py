import jax
import jax.numpy as jnp

def test_q_bitcast():
    bq_sz = 64
    num_q_heads = 8
    head_dim = 512
    q_dtype = jnp.bfloat16
    q_packing = 2
    
    # Create dummy q in VMEM
    q_init = jax.random.normal(jax.random.PRNGKey(0), (bq_sz, num_q_heads, head_dim)).astype(q_dtype)
    
    # Simulate bq_x2_ref
    bq_x2_ref = q_init
    
    # Simulate load_q
    # 1. bitcast to uint32 (flat first, using numpy)
    import numpy as np
    # Convert bfloat16 to float16 or uint16 to view as uint32 on numpy
    # We can cast to uint16 first using lax
    bq_x2_u16 = jax.lax.bitcast_convert_type(bq_x2_ref, jnp.uint16)
    np_u16 = np.array(bq_x2_u16)
    np_u32 = np_u16.view(np.uint32)
    bq_x2_u32 = jnp.array(np_u32)
    print("bq_x2_u32 shape:", bq_x2_u32.shape)
    
    # 2. Slice
    q_ref = bq_x2_u32
    
    # 3. Reshape (to bq_sz * num_q_heads_per_q_packing, head_dim)
    num_q_heads_per_q_packing = num_q_heads // q_packing
    q_ref_reshaped = q_ref.reshape(bq_sz * num_q_heads_per_q_packing, head_dim)
    print("q_ref_reshaped shape:", q_ref_reshaped.shape)
    
    # 4. Bitcast back to q_dtype
    np_ref_reshaped = np.array(q_ref_reshaped)
    np_bf16_as_u16 = np_ref_reshaped.view(np.uint16)
    q_bf16_u16 = jnp.array(np_bf16_as_u16)
    q_bf16 = jax.lax.bitcast_convert_type(q_bf16_u16, jnp.bfloat16)
    print("q_bf16 shape:", q_bf16.shape)
    
    # 5. Reshape to final q shape
    q_final = q_bf16.reshape(bq_sz * num_q_heads, head_dim)
    print("q_final shape:", q_final.shape)
    
    # Compare with original flattened q
    q_orig_flat = q_init.reshape(bq_sz * num_q_heads, head_dim)
    diff = jnp.abs(q_orig_flat - q_final).max()
    print("Max absolute difference:", diff)

if __name__ == "__main__":
    test_q_bitcast()
