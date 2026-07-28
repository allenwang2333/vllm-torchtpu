import jax
import jax.numpy as jnp
import numpy as np

def test_rope_reconstruct():
    # 1. Create random bf16
    key = jax.random.PRNGKey(42)
    rope_orig = jax.random.normal(key, (64,)).astype(jnp.bfloat16)
    
    # 2. Convert to uint8 bytes (using bitcast)
    rope_u8 = jax.lax.bitcast_convert_type(rope_orig, jnp.uint8)
    
    # 3. Simulate reconstruction logic on CPU
    # Flatten to simulate VMEM layout
    rope_bytes = np.array(rope_u8).reshape(64, 2)
    byte0 = rope_bytes[:, 0].astype(np.uint32)
    byte1 = rope_bytes[:, 1].astype(np.uint32)
    
    rope_uint16 = byte0 | (byte1 << 8)
    rope_float32_bits = rope_uint16 << 16
    
    # Convert bits to float32
    rope_f32 = jnp.array(np.frombuffer(rope_float32_bits.tobytes(), dtype=np.float32))
    rope_bf16 = rope_f32.astype(jnp.bfloat16)
    
    # Check difference
    diff = jnp.abs(rope_orig - rope_bf16).max()
    print("Orig:   ", rope_orig[:5])
    print("Recon:  ", rope_bf16[:5])
    print("Max Diff:", diff)

if __name__ == "__main__":
    test_rope_reconstruct()
