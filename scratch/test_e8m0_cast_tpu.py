import jax
import jax.numpy as jnp

def test_tpu_cast():
    # Powers of 2 scale factors
    sf = jnp.array([1.0, 2.0, 0.5, 0.25, 256.0, 0.0078125], dtype=jnp.float32)
    
    # Cast to e8m0
    e8m0 = sf.astype(jnp.float8_e8m0fnu)
    
    # Bitcast to uint8 to see the actual byte representation stored in memory
    u8 = jax.lax.bitcast_convert_type(e8m0, jnp.uint8)
    
    # Decode manually
    recon = jnp.exp2(u8.astype(jnp.float32) - 127.0)
    
    print("SF:    ", sf)
    print("Bytes: ", u8)
    print("Recon: ", recon)

if __name__ == "__main__":
    test_tpu_cast()
