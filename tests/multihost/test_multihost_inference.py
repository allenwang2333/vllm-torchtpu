import os
import sys

from vllm import LLM, SamplingParams


def main():
    print("Initializing Multi-Host LLM over Ray")

    # Initializing model with tensor parallelism
    try:
        llm = LLM(model="Qwen/Qwen2.5-0.5B",
                  tensor_parallel_size=8,
                  enforce_eager=True,
                  trust_remote_code=True,
                  load_format="runai_streamer")
    except Exception as e:
        print(f"Failed to initialize multi-host LLM: {e}")
        sys.exit(1)

    # Verifing distributed environments are correctly propagated
    assert "TORCH_TPU_TOPOLOGY" in os.environ, "Ray Distributed Executor failed to inject TORCH_TPU_TOPOLOGY"

    prompts = ["Ray multihost distributed serving on TorchTPU is"]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10)

    try:
        outputs = llm.generate(prompts, sampling_params)
        assert len(outputs) == 1
        generated_text = outputs[0].outputs[0].text
        print(f"Output: {generated_text}")
    except Exception as e:
        print(f"Collective communication or generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
