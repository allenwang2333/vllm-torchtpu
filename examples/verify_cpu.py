# Diagnostic script to run DeepSeek-V4 on CPU
import os
import sys

# Force vLLM to use CPU platform by overriding it before any engine initialization
import vllm.platforms
from vllm.platforms.cpu import CpuPlatform
vllm.platforms.current_platform = CpuPlatform()

from vllm import LLM, SamplingParams

def main():
    model_path = "examples/dsv4-flash-4layer-fp8"
    
    print(f"Initializing LLM on CPU with model: {model_path}...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        max_model_len=128,
        trust_remote_code=True,
    )
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=10,
    )
    
    prompt = "The quick brown fox jumps over the lazy dog"
    print(f"Generating for prompt: '{prompt}'...")
    outputs = llm.generate([prompt], sampling_params)
    
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_tokens = output.outputs[0].token_ids
        print(f"Prompt: {prompt!r}")
        print(f"Prompt token IDs: {output.prompt_token_ids}")
        print(f"Generated token IDs: {generated_tokens}")
        print(f"Generated text: {generated_text!r}")

if __name__ == "__main__":
    main()
