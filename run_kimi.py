from vllm import LLM, SamplingParams
import os

llm = LLM(
    model="/mnt/pds/xfgu-pd-v7x/kimik26/Kimi-K3-test",
    trust_remote_code=True,
    tensor_parallel_size=8,
    enforce_eager=False,
    disable_custom_all_reduce=True,
)

prompts = ["Hello, this is a test prompt."]
sampling_params = SamplingParams(max_tokens=10)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
