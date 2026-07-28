import os
import glob
import torch
from safetensors.torch import load_file

def check_all_safetensors():
    search_path = "/home/patemotter_google_com/workspace/hf_cache/hub/models--deepseek-ai--DeepSeek-V4/snapshots/*/*.safetensors"
    files = glob.glob(search_path)
    if not files:
        # Try finding in the huggingface cache directly
        search_path = "/home/patemotter_google_com/workspace/hf_cache/**/*.safetensors"
        files = glob.glob(search_path, recursive=True)
    
    print(f"Found {len(files)} safetensors files to check.")
    for f in sorted(files):
        try:
            weights = load_file(f, device="cpu")
            for name, tensor in weights.items():
                # Cast f8 to f32 first
                is_f8 = "float8" in str(tensor.dtype).lower()
                t_check = tensor.view(torch.uint8).to(torch.float32) if (is_f8 and "e8m0" in str(tensor.dtype).lower()) else (tensor.to(torch.float32) if is_f8 else tensor)
                if t_check.dtype.is_floating_point:
                    has_nan = torch.isnan(t_check).any().item()
                    has_inf = torch.isinf(t_check).any().item()
                    if has_nan or has_inf:
                        print(f"[NAN_FOUND] {f} -> {name}: shape={list(tensor.shape)} dtype={tensor.dtype} has_nan={has_nan} has_inf={has_inf}")
        except Exception as e:
            print(f"Error checking {f}: {e}")

if __name__ == "__main__":
    check_all_safetensors()
