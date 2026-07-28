# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Quick verification script for DeepSeek-V4 on TPU."""

import os
os.environ["NEW_MODEL_DESIGN"] = "1"
os.environ["MODEL_IMPL_TYPE"] = "vllm"
os.environ["VLLM_ENFORCE_EAGER"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
print("DEBUG_ENV_GPU_MEM_UTIL:", os.getenv("GPU_MEM_UTIL"))

import sys
import importlib.util

# Dynamic hot-patch to import our local patched pallas module before any vllm imports occur.
# This registers output_partition_specs support on GKE without modifying the container site-packages.
try:
    patch_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "vllm_torchtpu", "patches", "pallas_patched.py")
    if os.path.exists(patch_path):
        # 1. Import original module and submodule
        import torch_tpu._internal.pallas as original_pallas
        import torch_tpu._internal.pallas.pallas as original_pallas_submodule
        
        # 2. Load our patched module as a temporary standalone module
        spec = importlib.util.spec_from_file_location("patched_pallas_temp", patch_path)
        patched_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(patched_module)
        
        # 3. Patch both package and submodule namespaces
        for target in [original_pallas, original_pallas_submodule]:
            target.jax_op = patched_module.jax_op
            target.custom_jax_kernel = patched_module.custom_jax_kernel
            target.JaxCallable = patched_module.JaxCallable
            target.torch_placeholder = patched_module.torch_placeholder
            target._convert_shape = patched_module._convert_shape
        print("Successfully applied dynamic Pallas patches for output_partition_specs.")
except Exception as e:
    print(f"Failed to apply dynamic Pallas patches: {e}")

from vllm import LLM, SamplingParams


def main():
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.getenv("DSV4_MODEL_PATH", os.path.join(script_dir, "dsv4-flash-4layer-fp8"))
    
    print(f"Initializing LLM with model: {model_path}...")
    tp_size = int(os.getenv("TP_SIZE", "8"))
    ep_size = 8 // tp_size
    print(f"Configuring mesh with tp_size={tp_size}, ep_size={ep_size}...")
    llm = LLM(
        model=model_path,
        quantization="deepseek_v4_fp8",
        load_format="auto",
        tensor_parallel_size=tp_size,
        max_model_len=128,
        max_num_batched_tokens=128,
        trust_remote_code=True,
        kv_cache_dtype="fp8",
        gpu_memory_utilization=float(os.getenv("GPU_MEM_UTIL", "0.15")),
        enforce_eager=(os.getenv("VLLM_ENFORCE_EAGER", "0") == "1"),
        enable_expert_parallel=True,
        safetensors_load_strategy="lazy",
        compilation_config={
            "compile_sizes": [128],
        },
        additional_config={
            "sharding": {
                "sharding_strategy": {
                    "enable_dp_attention": True,
                    "expert_parallelism": ep_size,
                }
            }
        },
    )
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=50,
        logprobs=5,
        prompt_logprobs=5,
    )
    
    try:
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        print("[DEBUG] Listing loaded parameter statistics for fused_wkv_wgate:")
        for name, p in model.named_parameters():
            if "fused_wkv_wgate" in name:
                print(f"[DEBUG_LOADED_WEIGHT] {name}: shape={list(p.shape)} mean={p.mean().item():.6f} std={p.std().item():.6f}", flush=True)
    except Exception as e:
        print(f"[DEBUG] Failed to print model parameters: {e}")

    prompt = "Explain the difference between JAX and PyTorch in one sentence."
    print(f"Generating for prompt: '{prompt}'...")
    outputs = llm.generate([prompt], sampling_params)
    
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_tokens = output.outputs[0].token_ids
        print(f"Prompt: {prompt!r}")
        print(f"Prompt token IDs: {output.prompt_token_ids}")
        print(f"Prompt logprobs (last token): {output.prompt_logprobs[-1] if output.prompt_logprobs else 'None'}")
        print(f"Generated token IDs: {generated_tokens}")
        print(f"Generated logprobs (step 1): {output.outputs[0].logprobs[0] if output.outputs[0].logprobs else 'None'}")
        print(f"Generated text: {generated_text!r}")

if __name__ == "__main__":
    main()
