import argparse
import os
import time

import vllm_torchtpu  # noqa: F401

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"

from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.multimodal.image import convert_image_mode

# Enable compilation cache
os.environ["VLLM_XLA_CACHE_PATH"] = "/tmp/vllm_xla_cache"

MODEL_ID = "google/gemma-4-31B-it"


# Initialize vLLM with enforce_eager=True for TPU stability
# 31B uses TP=8 across all 8 cores of a 4-chip tpu7x-4 / v6e node.
def main():
    parser = argparse.ArgumentParser(
        description="Gemma 4 31B Text and Multimodal Inference Example on TPU")
    parser.add_argument(
        "--multimodal",
        action="store_true",
        help="Run multimodal (vision + text) inference instead of text-only.",
    )
    args = parser.parse_args()

    tp_size = int(os.environ.get("TP_SIZE", "8"))

    print(
        f"Initializing vLLM with model: {MODEL_ID}, TP={tp_size}, multimodal={args.multimodal}"
    )
    t_start = time.time()
    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=tp_size,
        enforce_eager=True,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.75,
        limit_mm_per_prompt={"image": 1} if args.multimodal else None,
    )
    t_init_end = time.time()

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.95,
        max_tokens=128,
    )

    if args.multimodal:
        print(f"Loading processor for {MODEL_ID}...")
        processor = AutoProcessor.from_pretrained(MODEL_ID,
                                                  trust_remote_code=True)

        print("Creating sample images...")
        image_red = Image.new("RGB", (224, 224), color="red")
        image_blue = Image.new("RGB", (224, 224), color="blue")
        image_cherry = convert_image_mode(
            ImageAsset("cherry_blossom").pil_image, "RGB")

        messages_list = [
            (
                [
                    {
                        "role":
                        "user",
                        "content": [
                            {
                                "type": "image"
                            },
                            {
                                "type": "text",
                                "text": "What color is this image?"
                            },
                        ],
                    },
                ],
                image_red,
            ),
            (
                [
                    {
                        "role":
                        "user",
                        "content": [
                            {
                                "type": "image"
                            },
                            {
                                "type":
                                "text",
                                "text":
                                "Describe this image and identify its dominant color.",
                            },
                        ],
                    },
                ],
                image_blue,
            ),
        ]

        img_questions = [
            "What is the content of this image?",
            "Describe the content of this image in detail.",
            "What's in the image?",
            "Where is this image taken?",
        ]
        for question in img_questions:
            messages_list.append((
                [
                    {
                        "role":
                        "user",
                        "content": [
                            {
                                "type": "image"
                            },
                            {
                                "type": "text",
                                "text": question
                            },
                        ],
                    },
                ],
                image_cherry,
            ))

        inputs = [{
            "prompt":
            processor.apply_chat_template(messages,
                                          tokenize=False,
                                          add_generation_prompt=True),
            "multi_modal_data": {
                "image": img
            },
        } for messages, img in messages_list]

        print("Running multimodal inference...")
        t_gen_start = time.time()
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        t_gen_end = time.time()
    else:
        print(f"Loading tokenizer for {MODEL_ID}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID,
                                                  trust_remote_code=True)

        messages_list = [
            [{
                "role": "user",
                "content": "Write a short poem about TPUs."
            }],
            [{
                "role": "user",
                "content": "What is the capital of France?"
            }],
        ]

        prompts = [
            tokenizer.apply_chat_template(messages,
                                          tokenize=False,
                                          add_generation_prompt=True)
            for messages in messages_list
        ]

        print("Running text-only inference...")
        t_gen_start = time.time()
        outputs = llm.generate(prompts, sampling_params=sampling_params)
        t_gen_end = time.time()

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print("-" * 50)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
        print("-" * 50)

    gen_duration = t_gen_end - t_gen_start
    total_duration = t_gen_end - t_start
    init_duration = t_init_end - t_start

    print("=" * 60)
    print("  PERFORMANCE METRICS SUMMARY")
    print(f"  - Engine Initialization Time: {init_duration:.2f} s")
    print(
        f"  - Generation Time ({len(outputs)} prompts): {gen_duration:.2f} s")
    print(
        f"  - Generation Throughput: {len(outputs) / gen_duration:.2f} prompts/s"
    )
    print(f"  - Total End-to-End Duration: {total_duration:.2f} s")
    print("=" * 60)


if __name__ == "__main__":
    main()
