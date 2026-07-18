import os
import torch
from transformers import AutoConfig

model_path = "/mnt/workspace/dsv4-flash-4layer-real-fp8"
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
print("Model Config:")
print(f"num_hidden_layers: {config.num_hidden_layers}")
print(f"hidden_size: {config.hidden_size}")
print(f"moe_intermediate_size: {config.moe_intermediate_size}")
print(f"n_routed_experts: {config.n_routed_experts}")

# Let's inspect the weights in the first shard to see the parameter shapes
shard_path = os.path.join(model_path, "model-00001-of-00046.safetensors")
# Let's inspect the safetensors index to find keys related to experts
index_path = os.path.join(model_path, "model.safetensors.index.json")
if os.path.exists(index_path):
    import json
    with open(index_path, "r") as f:
        index_data = json.load(f)
    print("\nFFN Weight Keys in Index:")
    weight_map = index_data.get("weight_map", {})
    count = 0
    for k in sorted(weight_map.keys()):
        if "ffn" in k:
            print(f"{k}: shard={weight_map[k]}")
            count += 1
            if count > 50:
                break
