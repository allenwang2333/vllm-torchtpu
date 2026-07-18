# Copyright 2026 Google LLC
import sys
import os
print("DEBUG_MOE SYS.PATH:", sys.path, flush=True)
print("DEBUG_MOE ENV:", dict(os.environ), flush=True)
import torch
import jax
import jax.numpy as jnp
import numpy as np

# Set device to TPU
from vllm.platforms import current_platform
current_platform.device_type = "tpu"

from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import set_vllm_model_wrapper_context
from vllm_torchtpu.layers.vllm.fused_moe import fused_moe_gmm

def main():
    # Setup dummy inputs
    num_tokens = 4
    hidden_size = 4096
    intermediate_size = 2048
    num_experts = 256
    topk = 8

    # Create mesh
    from jax.sharding import Mesh
    devices = jax.devices()
    print("JAX Devices:", devices)
    # Create 3D mesh compatible with partition specs: ('attn_dp', 'expert', 'model')
    mesh = Mesh(np.array([devices[0]]).reshape(1, 1, 1), ('attn_dp', 'expert', 'model'))
    print("Mesh initialized:", mesh)

    # Create inputs on TPU
    device = torch.device("tpu")
    print("TPU device:", device)

    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    # Packed weights shape
    w1 = torch.randint(0, 255, (num_experts, hidden_size, intermediate_size), dtype=torch.uint8, device=device)
    w2 = torch.randint(0, 255, (num_experts, intermediate_size, hidden_size // 2), dtype=torch.uint8, device=device)

    # Scale factors in e8m0 format (127 = 1.0)
    # Expected shape: (size_group, num_quant_blocks, 1, size_n)
    w1_scale = torch.full((num_experts, hidden_size // 32, 1, 2 * intermediate_size), 127, dtype=torch.uint8, device=device)
    w2_scale = torch.full((num_experts, intermediate_size // 32, 1, hidden_size), 127, dtype=torch.uint8, device=device)

    topk_weights = torch.full((num_tokens, topk), 1.0 / topk, dtype=torch.bfloat16, device=device)
    topk_ids = torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.int32, device=device)

    print("Running fused_moe_gmm...")
    with set_vllm_model_wrapper_context(mesh=mesh):
        try:
            out = fused_moe_gmm(
                hidden_states=x,
                w1=w1,
                w2=w2,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w1_bias=None,
                w2_bias=None,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                experts_start=0,
                topk=topk,
                activation="silu",
            )
            # Sync and print
            out_cpu = out.cpu()
            print("Output shape:", out_cpu.shape)
            print("Output values:", out_cpu)
            print("Contains NaN:", torch.isnan(out_cpu).any().item())
        except Exception as e:
            print("Exception during fused_moe_gmm execution:", e)
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
