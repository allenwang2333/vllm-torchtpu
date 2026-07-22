import os
import sys

# Set up environment variables
os.environ["NEW_MODEL_DESIGN"] = "1"
os.environ["MODEL_IMPL_TYPE"] = "vllm"

from vllm import LLM, SamplingParams

def main():
    llm = LLM(
        model="/root/cloud-devkit/torchtpu-vllm/examples/dsv4-flash-4layer-fp8",
        quantization="deepseek_v4_fp8",
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=1024,
        kv_cache_dtype="fp8",
        gpu_memory_utilization=0.05,
        enable_expert_parallel=False,
        enforce_eager=True,
        additional_config={
            "cache_config": {
                "num_gpu_blocks_override": 512,
            },
            "sharding": {
                "sharding_strategy": {
                    "enable_dp_attention": True,
                }
            }
        },
    )

    prompts = [
        "What is the speed of light in a vacuum?",
    ]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=5,
        ignore_eos=False,
        logprobs=5,
    )

    print("=== PyTorch TPU Verification Inference ===", flush=True)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        print(f"Prompt: {output.prompt!r}", flush=True)
        print(f"Prompt token IDs: {output.prompt_token_ids}", flush=True)
        print(f"Generated token IDs: {output.outputs[0].token_ids}", flush=True)
        print(f"Generated text: {output.outputs[0].text!r}", flush=True)
        if output.outputs[0].logprobs is not None and len(output.outputs[0].logprobs) > 0:
            print(f"Generated logprobs (step 1): {output.outputs[0].logprobs[0]}", flush=True)
        print("-" * 80, flush=True)

if __name__ == "__main__":
    main()
