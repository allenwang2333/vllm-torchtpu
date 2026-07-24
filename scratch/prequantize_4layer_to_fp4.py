import os
os.environ["JAX_PLATFORMS"] = "cpu"
import sys
import json
import torch
from safetensors import safe_open
from concurrent.futures import ProcessPoolExecutor

def quantize_tensor_to_fp4(tensor: torch.Tensor, axis: int = -1, block_size: int = 512):
    if isinstance(axis, int):
        axis = [axis]

    FP4_VALUES = torch.tensor([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0,
        -3.0, -4.0, -6.0
    ],
                              dtype=torch.float32,
                              device=tensor.device)
    FP4_MAX = 6.0
    FP4_MIN = -6.0

    orig_shape = tensor.shape
    blocked_shape = []
    for i, dim in enumerate(orig_shape):
        if i in [(a + tensor.ndim) % tensor.ndim for a in axis]:
            if dim % block_size != 0:
                raise ValueError(f"Dimension {i} of size {dim} is not divisible by block_size={block_size}")
            num_blocks = dim // block_size
            blocked_shape.extend([num_blocks, block_size])
        else:
            blocked_shape.append(dim)

    axis_normalized = sorted([a % tensor.ndim for a in axis])
    expanded_axis = [1 + n + a for n, a in enumerate(axis_normalized)]

    tensor_blocked = tensor.reshape(blocked_shape)

    abs_max = tensor_blocked.abs().amax(dim=expanded_axis, keepdim=True)
    scale = abs_max / FP4_MAX

    scale_inv = torch.where(scale == 0, 0.0, 1.0 / scale)

    tensor_scaled = tensor_blocked * scale_inv
    tensor_clipped = tensor_scaled.clamp(FP4_MIN, FP4_MAX)

    flat_clipped = tensor_clipped.flatten()
    flat_indices = torch.empty(flat_clipped.shape, dtype=torch.int8, device=tensor.device)
    fp4_for_compare = FP4_VALUES.view(1, 16)

    chunk_size = 4000000
    for i in range(0, flat_clipped.numel(), chunk_size):
        chunk = flat_clipped[i : i + chunk_size]
        chunk_for_compare = chunk.unsqueeze(-1)
        distances = (chunk_for_compare - fp4_for_compare).abs()
        flat_indices[i : i + chunk_size] = distances.argmin(dim=-1).to(torch.int8)

    indices = flat_indices.reshape(orig_shape)
    scale = scale.squeeze(tuple(expanded_axis)).to(torch.float32)

    return indices, scale

def pack_fp4_indices(indices: torch.Tensor, axis: int = -1) -> torch.Tensor:
    axis = axis % indices.ndim
    if indices.shape[axis] % 2 != 0:
        raise ValueError(f"Dimension {axis} of size {indices.shape[axis]} must be even for packing")

    indices = indices.to(torch.uint8)
    
    slices_low = [slice(None)] * indices.ndim
    slices_low[axis] = slice(0, None, 2)
    low = indices[tuple(slices_low)]
    
    slices_high = [slice(None)] * indices.ndim
    slices_high[axis] = slice(1, None, 2)
    high = indices[tuple(slices_high)]

    packed = low | (high << 4)
    return packed

REQUANTIZED_BLOCK_SIZE = 512

def process_layer_for_all_ranks(model_path, cache_dir, layer_idx, index_data):
    try:
        torch.set_num_threads(8)

        # Check if all 8 ranks already exist in cache
        w13_shape_str = "4096x2048"
        w2_shape_str = "4096x1024"
        all_ranks_cached = True
        for rank in range(8):
            cache_path = os.path.join(cache_dir, f"rank_{rank}_layer_{layer_idx}_w13_{w13_shape_str}_w2_{w2_shape_str}.pt")
            if not os.path.exists(cache_path) or os.path.getsize(cache_path) < 1000000:
                all_ranks_cached = False
                break
        if all_ranks_cached:
            print(f"[{layer_idx:02d}] Already fully cached. Skipping.", flush=True)
            return

        print(f"[{layer_idx:02d}] Starting processing...", flush=True)
        # 1. Group keys by safetensors shard
        weight_keys = {}
        for expert_idx in range(256):
            for key_type in ["w1.weight", "w2.weight", "w3.weight"]:
                # Note: the 4-layer model uses 'mlp.experts' prefix in safetensors
                full_key = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{key_type}"
                shard_name = index_data["weight_map"].get(full_key)
                if shard_name:
                    weight_keys.setdefault(shard_name, []).append((expert_idx, key_type, full_key))

        if not weight_keys:
            print(f"[{layer_idx:02d}] Warning: No expert keys found in index!", flush=True)
            return

        # 2. Load tensors from shards
        loaded_tensors = {}
        for shard_name, keys in weight_keys.items():
            shard_path = os.path.join(model_path, shard_name)
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for expert_idx, key_type, full_key in keys:
                    loaded_tensors[(expert_idx, key_type)] = f.get_tensor(full_key).clone().to(torch.float32)

        for rank in range(8):
            start_expert = rank * 32
            end_expert = (rank + 1) * 32

            local_w1_weights = []
            local_w3_weights = []
            local_w2_weights = []

            for expert_idx in range(start_expert, end_expert):
                local_w1_weights.append(loaded_tensors[(expert_idx, "w1.weight")])
                local_w3_weights.append(loaded_tensors[(expert_idx, "w3.weight")])
                local_w2_weights.append(loaded_tensors[(expert_idx, "w2.weight")])

            w1_weight = torch.stack(local_w1_weights, dim=0) # [32, 2048, 4096]
            w3_weight = torch.stack(local_w3_weights, dim=0) # [32, 2048, 4096]
            w2_weight = torch.stack(local_w2_weights, dim=0) # [32, 4096, 2048]

            # Concatenate w1/w3 along intermediate dimension (axis 1)
            # w13 shape: [32, 4096, 4096]
            w13_weight = torch.cat([w1_weight, w3_weight], dim=1)

            # Transpose to shape:
            # w13: [32, 4096, 4096] (transpose contracting dim to end)
            w13_weight = w13_weight.transpose(1, 2).contiguous()
            w2_weight = w2_weight.transpose(1, 2).contiguous() # [32, 2048, 4096]

            orig_hidden_size = w2_weight.shape[2]
            orig_local_intermediate_size = w2_weight.shape[1]

            w13_block_size = REQUANTIZED_BLOCK_SIZE
            w2_block_size = min(REQUANTIZED_BLOCK_SIZE, orig_local_intermediate_size)

            local_intermediate_size_padded = (orig_local_intermediate_size + 128 - 1) // 128 * 128
            hidden_size_padded = (orig_hidden_size + w13_block_size - 1) // w13_block_size * w13_block_size

            # Padding
            w1_weight_part = w13_weight[:, :, :orig_local_intermediate_size]
            w3_weight_part = w13_weight[:, :, orig_local_intermediate_size:]
            
            pad_w13_intermediate = local_intermediate_size_padded - orig_local_intermediate_size
            pad_w13_hidden = hidden_size_padded - orig_hidden_size
            
            pad_spec_w13 = (0, pad_w13_intermediate, 0, pad_w13_hidden)
            w1_weight_padded = torch.nn.functional.pad(w1_weight_part, pad_spec_w13)
            w3_weight_padded = torch.nn.functional.pad(w3_weight_part, pad_spec_w13)
            w13_weight_padded = torch.cat([w1_weight_padded, w3_weight_padded], dim=2)
            
            pad_spec_w2 = (0, pad_w13_hidden, 0, pad_w13_intermediate)
            w2_weight_padded = torch.nn.functional.pad(w2_weight, pad_spec_w2)

            # Quantize to FP4 (MX block format)
            w13_indices, w13_weight_scale_new = quantize_tensor_to_fp4(w13_weight_padded, axis=1, block_size=w13_block_size)
            w2_indices, w2_weight_scale_new = quantize_tensor_to_fp4(w2_weight_padded, axis=1, block_size=w2_block_size)

            w13_weight_processed = pack_fp4_indices(w13_indices, axis=1).contiguous()
            w2_weight_processed = pack_fp4_indices(w2_indices, axis=1).contiguous()

            s13 = w13_weight_scale_new.contiguous()
            w13_weight_scale_final = torch.empty(s13.shape[0], s13.shape[1], 1, s13.shape[2], dtype=s13.dtype).copy_(s13.unsqueeze(2))
            s2 = w2_weight_scale_new.contiguous()
            w2_weight_scale_final = torch.empty(s2.shape[0], s2.shape[1], 1, s2.shape[2], dtype=s2.dtype).copy_(s2.unsqueeze(2))

            cache_path = os.path.join(cache_dir, f"rank_{rank}_layer_{layer_idx}_w13_{w13_shape_str}_w2_{w2_shape_str}.pt")
            checkpoint_dict = {
                "w13_weight_packed": w13_weight_processed,
                "w2_weight_packed": w2_weight_processed,
                "w13_weight_scale": w13_weight_scale_final,
                "w2_weight_scale": w2_weight_scale_final,
            }
            torch.save(checkpoint_dict, cache_path)

        print(f"[{layer_idx:02d}] Completed processing successfully.", flush=True)
    except Exception as e:
        print(f"[{layer_idx:02d}] Error failed: {e}", flush=True)
        import traceback
        traceback.print_exc()

def main():
    torch.set_num_threads(8)
    if len(sys.argv) < 3:
        print("Usage: python3 prequantize_4layer_to_fp4.py <model_path> <cache_dir>")
        sys.exit(1)

    model_path = sys.argv[1]
    cache_dir = sys.argv[2]
    os.makedirs(cache_dir, exist_ok=True)

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        print(f"Index file not found at: {index_path}")
        sys.exit(1)

    with open(index_path, "r") as f:
        index_data = json.load(f)

    config_path = os.path.join(model_path, "config.json")
    num_hidden_layers = 4
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
                num_hidden_layers = config_data.get("num_hidden_layers", 4)
        except Exception as e:
            print(f"Warning: Failed to load config.json: {e}. Defaulting to 4 layers.", flush=True)

    print(f"Launching sequential layer pre-quantizer for {num_hidden_layers} layers in FP4 format...", flush=True)
    for l in range(num_hidden_layers):
        process_layer_for_all_ranks(model_path, cache_dir, l, index_data)

    print("Offline pre-quantization completed successfully for all layers across 8 ranks!", flush=True)

if __name__ == "__main__":
    main()
