import os
os.environ["NEW_MODEL_DESIGN"] = "1"
os.environ["MODEL_IMPL_TYPE"] = "vllm"

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
        sys.modules["patched_pallas_temp"] = patched_module
        spec.loader.exec_module(patched_module)

        
        # 3. Patch both package and submodule namespaces
        for target in [original_pallas, original_pallas_submodule]:
            target.jax_op = patched_module.jax_op
            target.custom_jax_kernel = patched_module.custom_jax_kernel
            target.JaxCallable = patched_module.JaxCallable
            target.torch_placeholder = patched_module.torch_placeholder
            target._convert_shape = patched_module._convert_shape
            
        print(f"=== [HOT PATCH] Successfully hot-patched pallas package and submodule from {patch_path} ===")
except Exception as e:
    print(f"=== [HOT PATCH ERROR] Failed to hot-patch pallas.py: {e} ===")


from vllm import LLM, SamplingParams


def main():
    # Instantiate the LLM. It will automatically use our TPU platform and patches.
    # We use TP=8 to utilize all 8 TPU chips on the v7x-8 node.
    llm = LLM(
        model=os.getenv("DSV4_MODEL_PATH", "deepseek-ai/DeepSeek-V4-Flash"),
        quantization="deepseek_v4_fp8",
        tensor_parallel_size=8,
        trust_remote_code=True,
        max_model_len=64,
        max_num_batched_tokens=64,
        kv_cache_dtype="fp8",
        gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.20")),
        enable_expert_parallel=True,
        enforce_eager=os.getenv("VLLM_ENFORCE_EAGER", "0") == "1",
        additional_config={

            "compilation_config": {
                "compile_sizes": [64],
            },
            "cache_config": {
                "num_gpu_blocks_override": 512,
            },
            "sharding": {
                "sharding_strategy": {
                    "enable_dp_attention": False,
                    "expert_parallelism": 8,
                }
            }
        },
    )

    prompts = [
        "What is the speed of light in a vacuum?",
    ]

    sampling_params = SamplingParams(temperature=0.0, max_tokens=5, ignore_eos=False)

    print("=== Testing raw and manually formatted prompts with llm.generate (ignore_eos=True) ===")
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_tokens = output.outputs[0].token_ids
        print(f"Prompt: {prompt!r}")
        print(f"Generated token IDs: {generated_tokens}")
        print(f"Generated text: {generated_text!r}")
        print("-" * 80)

    print("=== Testing chat API with llm.chat (ignore_eos=True) ===")
    messages = [
        [{"role": "user", "content": "Explain the theory of relativity in simple terms."}],
    ]
    chat_outputs = llm.chat(messages, sampling_params)
    for output in chat_outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_tokens = output.outputs[0].token_ids
        print(f"Chat Prompt: {prompt!r}")
        print(f"Generated token IDs: {generated_tokens}")
        print(f"Generated text: {generated_text!r}")
        print("-" * 80)

if __name__ == "__main__":
    main()
