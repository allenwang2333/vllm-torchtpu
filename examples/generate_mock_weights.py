# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0
import os
import json
import torch



from safetensors.torch import save_file

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.getenv("MOCK_WEIGHTS_DIR", os.path.join(script_dir, "dsv4-flash-4layer-fp8"))
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    complete_flag = os.path.join(model_dir, "mock_weights.complete")
    
    worker_id = int(os.getenv("RANK", "0"))
    if worker_id != 0:
        print(f"Worker {worker_id}: Waiting for worker 0 to generate mock weights...")
        import time
        while not os.path.exists(complete_flag):
            time.sleep(2)
        print(f"Worker {worker_id}: Mock weights ready, proceeding.")
        return

    # Check if files already exist to skip regeneration
    if os.path.exists(complete_flag):
        print("Mock safetensors weights already exist. Skipping generation.")
        return

    print("Generating mock safetensors weights locally on the host VM...")
    with open(index_path, "r") as f:
        index_data = json.load(f)

    weight_map = index_data["weight_map"]
    shards = {}
    for k, filename in weight_map.items():
        if filename not in shards:
            shards[filename] = []
        shards[filename].append(k)

    shapes_map_path = os.path.join(model_dir, "shapes_map.json")
    if not os.path.exists(shapes_map_path):
        raise FileNotFoundError(f"shapes_map.json not found at {shapes_map_path}. Please sync it first.")

    with open(shapes_map_path, "r") as f:
        shapes_map = json.load(f)
    print(f"Loaded {len(shapes_map)} parameter shapes from shapes_map.json.")

    dtype_map = {
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        "F32": torch.float32,
        "UI8": torch.uint8,
        "F8_E4M3": torch.float8_e4m3fn,
    }

    for filename, keys in shards.items():
        out_path = os.path.join(model_dir, filename)
        print(f"Building mock safetensors shard: {filename}...")
        tensor_dict = {}
        for k in keys:
            shape = None
            dtype = torch.float16
            
            if k in shapes_map:
                shape = shapes_map[k]["shape"]
                dtype_str = shapes_map[k]["dtype"]
                dtype = dtype_map.get(dtype_str, torch.float16)
            
                if "scale" in k:
                    # Scale factors in e8m0 format (127.0 corresponds to exponent 0 -> multiplier 1.0)
                    t = torch.full(shape, 127.0, dtype=dtype)
                    print(f"[DEBUG_GENERATE_SCALE] Generated scale key {k} with shape {shape} dtype {dtype}: Mean={t.mean():.1f}", flush=True)
                else:
                    if dtype == torch.float8_e4m3fn:
                        # uniform_ is not supported directly on float8, so generate in bf16 and cast
                        t = torch.empty(shape, dtype=torch.bfloat16).uniform_(-0.1, 0.1).to(dtype)
                    else:
                        t = torch.empty(shape, dtype=dtype).uniform_(-0.1, 0.1)
                tensor_dict[k] = t
            else:
                tensor_dict[k] = torch.ones([1], dtype=torch.float32)
                
        save_file(tensor_dict, out_path)
        print(f"Successfully saved {out_path} ({os.path.getsize(out_path) / (1024**2):.2f} MB)")

    with open(complete_flag, "w") as f:
        f.write("complete")
    print("Mock weights generation completed successfully.")

if __name__ == "__main__":
    main()
