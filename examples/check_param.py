import sys
import os
sys.argv = [sys.argv[0], "--step", "2"]
print("Initial sys.path:", sys.path, flush=True)
sys.path.insert(0, "/root/cloud-devkit/torchtpu-vllm/src")
sys.path.insert(0, "/root/cloud-devkit/torchtpu-vllm")
print("Updated sys.path:", sys.path, flush=True)
try:
    print("Listing /root/cloud-devkit/torchtpu-vllm:", os.listdir("/root/cloud-devkit/torchtpu-vllm"), flush=True)
    print("Listing /root/cloud-devkit/torchtpu-vllm/src:", os.listdir("/root/cloud-devkit/torchtpu-vllm/src"), flush=True)
except Exception as e:
    print("Listing failed:", e, flush=True)

from vllm.config import VllmConfig
from vllm_torchtpu.models.vllm.deepseek_v4_patch import _maybe_patch_for_deepseek_v4
from vllm.model_executor.model_loader import get_model
import torch
import gc
from examples.verify_pipeline_parity import make_mock_config

def main():
    vc = make_mock_config(device="tpu")
    gc.collect()
    with _maybe_patch_for_deepseek_v4(vc):
        model = get_model(vllm_config=vc)
        experts = model.model.layers[0].ffn.experts
        print("w13_weight:", type(experts.w13_weight), experts.w13_weight.shape, getattr(experts.w13_weight, "weight_loader", None))
        print("w13_weight_scale:", type(experts.w13_weight_scale), experts.w13_weight_scale.shape, getattr(experts.w13_weight_scale, "weight_loader", None))

if __name__ == "__main__":
    main()
