import jax
import jax.numpy as jnp

def test_fp8_edge_cases():
    # Test casting large float to float8_e4m3fn
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    deltas = [0.0, 0.0001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0]
    vals = jnp.array([fp8_max + d for d in deltas], dtype=jnp.float32)
    
    fp8_vals = vals.astype(jnp.float8_e4m3fn)
    recon_vals = fp8_vals.astype(jnp.float32)
    
    for v, r in zip(vals, recon_vals):
        print(f"Val: {v:<12} -> Recon: {r:<12} (is_nan={jnp.isnan(r)})")

if __name__ == "__main__":
    test_fp8_edge_cases()
