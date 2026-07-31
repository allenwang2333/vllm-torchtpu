from vllm import LLM, SamplingParams

model_path = "/mnt/pds/xfgu-pd-v7x/kimik26/Kimi-K3-test"

llm = LLM(
    model=model_path,
    trust_remote_code=True,
    tensor_parallel_size=8,
    gpu_memory_utilization=0.6,
    disable_log_stats=True,
    disable_custom_all_reduce=True,
    compilation_config={"compile_sizes": [16]},
)

prompts = ["Hello, this is a test prompt."]
sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

outputs = llm.generate(prompts, sampling_params)

for out in outputs:
    print(f"Prompt: {out.prompt!r}")
    out0 = out.outputs[0]
    print(f"Generated text: {out0.text!r}")
    print(f"Token IDs: {out0.token_ids}")
    print(f"Finish reason: {out0.finish_reason!r}")
