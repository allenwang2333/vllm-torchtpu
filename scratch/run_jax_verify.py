# Copyright 2026 Google LLC
# Wrapper to run JAX reference model with backend patch

import os
import sys

os.environ["XLA_CLIENT_MEM_FRACTION"] = "0.32"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["VLLM_ENFORCE_EAGER"] = "1"

# Add torchtpu-vllm/src to path so we can import env_override
sys.path.insert(0, "/root/cloud-devkit/vllm-torchtpu/src")
import vllm_torchtpu.env_override

# Now add our repo to path and import offline_inference
sys.path.insert(0, "/root/cloud-devkit/vllm-torchtpu")
from examples.offline_inference import create_parser
from vllm import LLM, SamplingParams

def custom_main(args: dict):
    # Pop arguments not used by LLM
    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    top_k = args.pop("top_k")
    use_chat_template = args.pop("use_chat_template")
    chat_template_kwargs = args.pop('chat_template_kwargs')

    llm = LLM(**args)

    sampling_params = llm.get_default_sampling_params()
    sampling_params.logprobs = 5
    sampling_params.max_tokens = 5
    if temperature is not None:
        sampling_params.temperature = temperature
    if top_p is not None:
        sampling_params.top_p = top_p
    if top_k is not None:
        sampling_params.top_k = top_k

    prompts = [
        "What is the speed of light in a vacuum?",
    ]

    print("Using custom main to dump logits and logprobs", flush=True)
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated = output.outputs[0]
        print(f"Prompt: {prompt!r}")
        print(f"Prompt token IDs: {output.prompt_token_ids}")
        print(f"Generated text: {generated.text!r}")
        print(f"Generated token IDs: {generated.token_ids}")
        if generated.logprobs is not None and len(generated.logprobs) > 0:
            print(f"Generated logprobs (step 1): {generated.logprobs[0]}")
        print("-" * 50)

if __name__ == "__main__":
    os.environ.setdefault('SKIP_JAX_PRECOMPILE', '1')
    parser = create_parser()
    args = vars(parser.parse_args())
    custom_main(args)
