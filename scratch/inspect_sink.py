import torch
import numpy as np

path = "/home/patemotter_google_com/workspace/vllm-torchtpu/scratch/nan_dump_model.layers.0.attn_rank0.pt"
print(f"Loading {path}...")
data = torch.load(path)

for k, v in data.items():
    if isinstance(v, torch.Tensor):
        has_nan = torch.isnan(v).any().item()
        has_inf = torch.isinf(v).any().item()
        print(f"Key: {k:25s} | Shape: {str(list(v.shape)):30s} | Dtype: {str(v.dtype):15s} | Has NaN: {has_nan} | Has Inf: {has_inf}")
        if v.numel() > 0 and v.dtype in (torch.float32, torch.bfloat16):
            v_f = v.float()
            clean_v = v_f[~torch.isnan(v_f) & ~torch.isinf(v_f)]
            if clean_v.numel() > 0:
                print(f"  Clean Stats: min={clean_v.min().item():.6f} | max={clean_v.max().item():.6f} | mean={clean_v.mean().item():.6f}")
    else:
        print(f"Key: {k:25s} | Type: {type(v)}")

# Detailed look at attn_sink
if "attn_sink" in data:
    sink = data["attn_sink"]
    print("attn_sink values:")
    print(sink.float().tolist())
