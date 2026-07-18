import os
import json
import torch
from safetensors import safe_open
from safetensors.torch import save_file

def main():
    real_model_dir = "/mnt/workspace/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1"
    output_dir = "/mnt/workspace/dsv4-flash-4layer-real-fp8"
    
    os.makedirs(output_dir, exist_ok=True)
    
    index_path = os.path.join(real_model_dir, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index_data = json.load(f)
        
    weight_map = index_data["weight_map"]
    
    # Group keys by shard file
    shard_to_keys = {}
    for k, filename in weight_map.items():
        if filename not in shard_to_keys:
            shard_to_keys[filename] = []
        shard_to_keys[filename].append(k)
        
    def should_keep(key):
        parts = key.split(".")
        if len(parts) > 1 and parts[0] == "layers":
            try:
                layer_idx = int(parts[1])
                return layer_idx < 4
            except (ValueError, IndexError):
                pass
        if len(parts) > 2 and parts[0] == "model" and parts[1] == "layers":
            try:
                layer_idx = int(parts[2])
                return layer_idx < 4
            except (ValueError, IndexError):
                pass
        # If it doesn't match layer patterns, it is a non-layer tensor (embed, head, norms), keep it.
        return True

    new_weight_map = {}
    
    for filename, keys in shard_to_keys.items():
        kept_keys = [k for k in keys if should_keep(k)]
        if not kept_keys:
            print(f"Skipping shard {filename} (no tensors to keep)")
            continue
            
        print(f"Processing shard {filename} (keeping {len(kept_keys)}/{len(keys)} tensors)...")
        in_shard_path = os.path.join(real_model_dir, filename)
        out_shard_path = os.path.join(output_dir, filename)
        
        tensor_dict = {}
        with safe_open(in_shard_path, framework="pt", device="cpu") as f:
            for k in kept_keys:
                tensor_dict[k] = f.get_tensor(k)
                new_weight_map[k] = filename
                
        save_file(tensor_dict, out_shard_path)
        print(f"Saved {out_shard_path}")
        
    # Write new index file
    new_index_data = {
        "metadata": index_data.get("metadata", {}),
        "weight_map": new_weight_map
    }
    new_index_path = os.path.join(output_dir, "model.safetensors.index.json")
    with open(new_index_path, "w") as f:
        json.dump(new_index_data, f, indent=2)
    print(f"Saved {new_index_path}")
    
    # Copy config files and modify layers count
    for conf_file in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"]:
        src = os.path.join(real_model_dir, conf_file)
        dst = os.path.join(output_dir, conf_file)
        if os.path.exists(src):
            import shutil
            shutil.copy(src, dst)
            print(f"Copied {conf_file}")
            
            if conf_file == "config.json":
                with open(dst, "r") as f:
                    config = json.load(f)
                config["num_hidden_layers"] = 4
                with open(dst, "w") as f:
                    json.dump(config, f, indent=2)
                print("Updated num_hidden_layers to 4 in config.json")

if __name__ == "__main__":
    main()
