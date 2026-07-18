import vllm_torchtpu # MUST be imported first to apply env/backend overrides!
import torch
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import torchax.ops.mappings as mappings
import numpy as np

# 1. IEEE FP4 values mapping
FP4_VALUES = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=torch.float32)

def quantize_to_fp4_indices(x: torch.Tensor) -> torch.Tensor:
    # Find closest FP4 value index
    diff = torch.abs(x.unsqueeze(-1) - FP4_VALUES)
    return torch.argmin(diff, dim=-1).to(torch.uint8)

def pack_fp4(indices: torch.Tensor) -> torch.Tensor:
    # Pack along axis 0: indices has shape (K, M)
    low = indices[0::2, ...]
    high = indices[1::2, ...]
    return low | (high << 4)

def test_flow():
    # Setup host sizes (must be multiples of 16/32 for TPU MXU alignment)
    M, K, N = 128, 128, 128
    
    print(f"Creating mock floating point weight matrix [{K}, {M}]...")
    # Structure weights as (K, M) so that K is axis 0 (aligned with JAX axis-0 bitcast scaling)
    w_raw = torch.randn(K, M)
    
    print("Quantizing weights to FP4 indices...")
    w_indices = quantize_to_fp4_indices(w_raw)
    
    print("Packing FP4 indices (two per byte) into uint8 matrix...")
    w_packed_cpu = pack_fp4(w_indices)
    
    # Dequantized weight mapping on CPU for golden reference
    w_dequant_cpu = FP4_VALUES[w_indices.to(torch.int64)]
    
    # Move operands to TPU device
    device = torch.device("tpu")
    print(f"Uploading packed uint8 weight matrix [{K//2}, {M}] to TPU...")
    w_packed = w_packed_cpu.to(device)
    
    y = torch.randn(K, N, dtype=torch.float32, device=device)
    
    # Golden reference uses w_dequant_cpu (K, M) transposed to (M, K) for matmul with y (K, N)
    print("Computing golden reference matrix multiplication in PyTorch...")
    golden_out = torch.matmul(w_dequant_cpu.to(device).T, y)
    
    # Bridge to JAX arrays using torchax's native t2j zero-copy translation
    print("Bridging PyTorch tensors to JAX arrays via torchax t2j...")
    jax_w_packed = mappings.t2j(w_packed)
    jax_y = mappings.t2j(y)
    
    # Define Pallas kernel function (takes references and writes to out_ref)
    def my_pallas_kernel(x_ref, y_ref, out_ref):
        # x_ref has shape (K//2, M) of type uint8.
        # pltpu.bitcast scales axis 0, yielding shape (K, M) of type float4_e2m1fn.
        x_fp4 = pltpu.bitcast(x_ref[...], jnp.float4_e2m1fn)
        # Convert float4 to float32, transpose to (M, K) to match contracting dimension of y
        out_ref[...] = jnp.matmul(x_fp4.astype(jnp.float32).T, y_ref[...])
    
    # Wrap it via pallas_call. BlockSpec parameters: BlockSpec(block_shape, index_map)
    run_pallas = pl.pallas_call(
        my_pallas_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
        grid=(1, 1),
        in_specs=[
            pl.BlockSpec((K // 2, M), lambda *args: (0, 0)),
            pl.BlockSpec((K, N), lambda *args: (0, 0)),
        ],
        out_specs=pl.BlockSpec((M, N), lambda *args: (0, 0))
    )
    
    print("Compiling and executing Pallas FP4-bitcast kernel on TPU VM...")
    pallas_out = run_pallas(jax_w_packed, jax_y)
    
    # Compare results
    pallas_out_torch = torch.from_numpy(np.array(pallas_out)).to(device)
    max_diff = torch.max(torch.abs(pallas_out_torch - golden_out))
    
    print(f"\nVerification successful!")
    print(f"* Pallas Output Max Absolute Diff vs. Golden: {max_diff.item():.6e}")
    assert max_diff < 1e-4, f"Output mismatch: {max_diff}"

if __name__ == "__main__":
    test_flow()
