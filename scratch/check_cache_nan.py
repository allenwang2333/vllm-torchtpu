import os
import torch

cache_dir = "/dev/shm/dsv4-flash-real-fp4"
if not os.path.exists(cache_dir):
    print(f"Cache dir {cache_dir} does not exist!")
    exit(1)

files = [f for f in os.listdir(cache_dir) if f.endswith(".pt")]
print(f"Found {len(files)} files in cache.")

nan_found = False
for f in sorted(files)[:20]: # Check first 20 files
    path = os.path.join(cache_dir, f)
    try:
        data = torch.load(path, map_location="cpu")
        print(f"Checking {f}...")
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                # If dtype is uint8, it's packed, can't contain floats/NaNs, but let's check scales which are float
                if v.dtype in (torch.float32, torch.bfloat16, torch.float16):
                    has_nan = torch.isnan(v).any().item()
                    if has_nan:
                        print(f"  NaN found in tensor {k} of file {f}!")
                        nan_found = True
    except Exception as e:
        print(f"  Error loading {f}: {e}")

if not nan_found:
    print("No NaNs found in the checked cache files.")
