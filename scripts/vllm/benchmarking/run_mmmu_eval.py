#!/usr/bin/env python3
"""
Standalone MMMU / MMMU-Pro Generative Evaluation Script for vLLM HTTP Server.

Evaluates multimodal vision-language models (e.g. Gemma 4 31B) via the OpenAI-compatible
/v1/chat/completions HTTP endpoint using generative multiple-choice QA (e.g. "Answer: (A)").

Technical Motivation:
1. Replaces lm_eval's log-likelihood scoring, which is unsupported over external HTTP API servers.
2. Avoids the official MMMU repo's offline PyTorch loader requirement (which would double-allocate
   TPU HBM alongside an active `vllm serve` process).
3. Benchmarks live vLLM HTTP REST endpoints directly using standard HuggingFace MMMU validation samples.
"""

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from typing import Dict, List, Tuple

import requests


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MMMU Generative Evaluation against vLLM HTTP Server")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument(
        "--model",
        default="google/gemma-4-31B-it",
        help="Model name registered in vLLM",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of samples to evaluate (default: 100)",
    )
    parser.add_argument(
        "--results-dir",
        default="/tmp/perf_eval_gemma-4-31b-it",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--dataset",
        default="MMMU/MMMU",
        help=
        "HuggingFace dataset to evaluate (default: 'MMMU/MMMU', supports 'MMMU/MMMU_Pro')",
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help=
        "Allow synthetic sample fallback if HuggingFace dataset fails to load",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help=
        "Run offline direct vLLM LLM engine generation instead of HTTP server",
    )
    return parser.parse_args()


def encode_image_base64(image) -> str:
    """Convert PIL Image to JPEG base64 string."""
    buffered = io.BytesIO()
    image.convert("RGB").save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def extract_answer(response_text: str) -> str:
    """Extract choice letter (A, B, C, D) from generated text."""
    # 1. Match explicit keyword patterns like "Answer: (A)", "Option A", "choice is D"
    match = re.search(
        r"\b(?:answer|option|choice)\b.*?\(?([A-D])\)?\b",
        response_text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()

    # 2. Match enclosed letter like "(B)", "A)", "[C]"
    match = re.search(r"(?:\(|\[|^)([A-D])(?:\)|\]|\b)", response_text)
    if match:
        return match.group(1).upper()

    # 3. Fallback: check for isolated uppercase single letter
    match = re.search(r"\b([A-D])\b", response_text)
    if match:
        return match.group(1).upper()

    return "UNKNOWN"


def load_mmmu_samples(
        dataset_name: str = "MMMU/MMMU",
        limit: int = 100,
        allow_synthetic: bool = False) -> Tuple[List[Dict], str, bool]:
    """Load MMMU or MMMU-Pro validation samples from HuggingFace datasets or fallback dataset."""
    try:
        from datasets import load_dataset

        print(
            f"Loading {dataset_name} dataset from HuggingFace (limit={limit})..."
        )
        samples = []
        if "MMMU_Pro" in dataset_name or "mmmu_pro" in dataset_name.lower():
            # MMMU-Pro dataset loading
            target_ds = "MMMU/MMMU_Pro"
            configs = ["standard", "vision"]
            for config in configs:
                if len(samples) >= limit:
                    break
                try:
                    ds = load_dataset(target_ds, config, split="test")
                except Exception:
                    try:
                        ds = load_dataset(target_ds,
                                          config,
                                          split="validation")
                    except Exception:
                        continue
                for item in ds:
                    if len(samples) >= limit:
                        break
                    image = item.get("image_1") or item.get(
                        "image") or item.get("img")
                    if image is None:
                        continue
                    question = item.get("question", "")
                    options_raw = item.get("options", [])
                    if isinstance(options_raw, str):
                        try:
                            import ast

                            options = ast.literal_eval(options_raw)
                        except Exception:
                            options = []
                    elif isinstance(options_raw, list):
                        options = options_raw
                    else:
                        options = []

                    answer = str(item.get("answer", "")).strip().upper()
                    samples.append({
                        "id":
                        item.get("id", f"mmmu_pro_{len(samples)}"),
                        "question":
                        question,
                        "options":
                        options,
                        "answer":
                        answer,
                        "image":
                        image,
                    })
        else:
            # Standard MMMU multi-domain dataset loading
            target_ds = "MMMU/MMMU"
            subjects = [
                "Accounting",
                "Finance",
                "Manage",
                "Marketing",
                "Economics",
                "Art_and_Design",
            ]
            for subject in subjects:
                if len(samples) >= limit:
                    break
                try:
                    dataset = load_dataset(target_ds,
                                           subject,
                                           split="validation")
                except Exception:
                    continue
                for item in dataset:
                    if len(samples) >= limit:
                        break
                    # Find image key (image_1, image, etc.)
                    image = item.get("image_1") or item.get("image")
                    if image is None:
                        continue
                    question = item.get("question", "")
                    options_raw = item.get("options", [])
                    if isinstance(options_raw, str):
                        try:
                            import ast

                            options = ast.literal_eval(options_raw)
                        except Exception:
                            options = []
                    elif isinstance(options_raw, list):
                        options = options_raw
                    else:
                        options = []

                    answer = str(item.get("answer", "")).strip().upper()

                    samples.append({
                        "id":
                        item.get("id", f"{subject}_{len(samples)}"),
                        "question":
                        question,
                        "options":
                        options,
                        "answer":
                        answer,
                        "image":
                        image,
                    })
        if not samples:
            raise RuntimeError(
                f"No valid samples loaded from HuggingFace {dataset_name} dataset."
            )
        print(
            f"Successfully loaded {len(samples)} HuggingFace {dataset_name} samples."
        )
        return samples, dataset_name, False
    except Exception as e:
        if not allow_synthetic:
            raise RuntimeError(
                f"Could not load HuggingFace {dataset_name} dataset ({e}). "
                "Specify --allow-synthetic to permit synthetic benchmark fallback."
            ) from e

        print(f"Warning: Could not load HuggingFace MMMU dataset ({e}).")
        print(
            "Falling back to synthetic multimodal benchmark suite (--allow-synthetic set)..."
        )
        from PIL import Image

        samples = []
        choices = ["A", "B", "C", "D"]
        colors = ["red", "blue", "green", "yellow"]
        for i in range(min(limit, 20)):
            color = colors[i % 4]
            img = Image.new("RGB", (224, 224), color=color)
            samples.append({
                "id":
                f"synthetic_{i}",
                "question":
                ("What is the dominant background color of this image?\n(A)"
                 " Red\n(B) Blue\n(C) Green\n(D) Yellow"),
                "options": ["Red", "Blue", "Green", "Yellow"],
                "answer":
                choices[i % 4],
                "image":
                img,
            })
        return samples, "synthetic_color_shapes", True


def evaluate_sample(sample: Dict, host: str, port: int,
                    model: str) -> Tuple[bool, str, str]:
    """Send a single multimodal request to vLLM /v1/chat/completions."""
    import re

    url = f"http://{host}:{port}/v1/chat/completions"
    img_b64 = encode_image_base64(sample["image"])

    question_clean = re.sub(r"<image \d+>", "", sample["question"]).strip()
    prompt_text = question_clean

    options = sample.get("options", [])
    if isinstance(options, list) and len(options) > 0:
        if "(A)" not in prompt_text and "A." not in prompt_text:
            prompt_text += "\n\nOptions:\n" + "\n".join(
                f"({chr(65+idx)}) {opt}" for idx, opt in enumerate(options))

    prompt_text += "\n\nRespond with only the final option letter in format (A), (B), (C), or (D)."

    payload = {
        "model":
        model,
        "messages": [{
            "role":
            "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}"
                    },
                },
                {
                    "type": "text",
                    "text": prompt_text
                },
            ],
        }],
        "max_tokens":
        2048,
        "temperature":
        0.0,
    }

    resp = requests.post(url, json=payload, timeout=300)
    if resp.status_code != 200:
        raise RuntimeError(
            f"HTTP Error {resp.status_code} for sample {sample['id']}: {resp.text[:500]}"
        )

    res_json = resp.json()
    generated_text = res_json["choices"][0]["message"]["content"]
    predicted = extract_answer(generated_text)
    is_correct = predicted == sample["answer"]
    return is_correct, predicted, sample["answer"]


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    print("====================================================")
    print("  MMMU / MMMU-Pro Generative Vision Evaluation")
    if args.offline:
        print("  Target: Offline vLLM LLM Engine (Direct TPU)")
    else:
        print(f"  Target: http://{args.host}:{args.port}/v1/chat/completions")
    print(f"  Model: {args.model} | Limit: {args.limit}")
    print("====================================================")

    samples, dataset_source, is_synthetic = load_mmmu_samples(
        args.dataset, args.limit, args.allow_synthetic)
    if not samples:
        print("ERROR: No samples loaded for MMMU evaluation.")
        sys.exit(1)

    correct_count = 0
    total_count = len(samples)
    results = []

    t0 = time.time()

    if args.offline:
        os.environ["VLLM_USE_V1"] = "0"
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        print("Initializing offline vLLM LLM engine on TPU (TP=8)...")
        processor = AutoProcessor.from_pretrained(args.model,
                                                  trust_remote_code=True)
        llm = LLM(
            model=args.model,
            tensor_parallel_size=8,
            gpu_memory_utilization=0.8,
            enforce_eager=True,
            trust_remote_code=True,
        )
        sampling_params = SamplingParams(max_tokens=2048, temperature=0.0)

        inputs = []
        for sample in samples:
            prompt_text = (
                f"{sample['prompt']}\n\n"
                "Please select the correct option by responding with ONLY the single letter corresponding to your choice (e.g. A, B, C, or D).\n"
                "Answer:")
            messages = [{
                "role":
                "user",
                "content": [
                    {
                        "type": "image"
                    },
                    {
                        "type": "text",
                        "text": prompt_text
                    },
                ],
            }]
            formatted_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs.append({
                "prompt": formatted_prompt,
                "multi_modal_data": {
                    "image": sample["image"]
                },
            })

        print(
            f"Running offline multimodal inference for {total_count} samples..."
        )
        outputs = llm.generate(inputs, sampling_params=sampling_params)

        for idx, (sample, output) in enumerate(zip(samples, outputs), 1):
            generated_text = output.outputs[0].text
            pred = extract_answer(generated_text)
            target = sample["answer"]
            is_correct = pred == target
            if is_correct:
                correct_count += 1
            results.append({
                "id": sample["id"],
                "predicted": pred,
                "ground_truth": target,
                "is_correct": is_correct,
            })
            if idx % 10 == 0 or idx == total_count:
                acc = correct_count / idx * 100.0
                print(f"[{idx}/{total_count}] Current Accuracy: {acc:.2f}%"
                      f" ({correct_count}/{idx})")
    else:
        for idx, sample in enumerate(samples, 1):
            is_correct, pred, target = evaluate_sample(sample, args.host,
                                                       args.port, args.model)
            if is_correct:
                correct_count += 1

            results.append({
                "id": sample["id"],
                "predicted": pred,
                "ground_truth": target,
                "is_correct": is_correct,
            })

            if idx % 10 == 0 or idx == total_count:
                acc = correct_count / idx * 100.0
                print(f"[{idx}/{total_count}] Current Accuracy: {acc:.2f}%"
                      f" ({correct_count}/{idx})")

    t1 = time.time()
    accuracy = (correct_count /
                total_count) * 100.0 if total_count > 0 else 0.0

    print("\n====================================================")
    print(f"  MMMU Evaluation Completed in {t1 - t0:.2f}s")
    print(f"  Dataset Source: {dataset_source} (is_synthetic={is_synthetic})")
    print(f"  Total Samples: {total_count}")
    print(
        f"  Exact Match Accuracy: {accuracy:.2f}% ({correct_count}/{total_count})"
    )
    print("====================================================")

    output_file = os.path.join(args.results_dir, "mmmu_eval_results.json")
    with open(output_file, "w") as f:
        json.dump(
            {
                "model": args.model,
                "dataset_source": dataset_source,
                "is_synthetic": is_synthetic,
                "samples": total_count,
                "accuracy": accuracy,
                "correct": correct_count,
                "duration_seconds": t1 - t0,
                "details": results,
            },
            f,
            indent=2,
        )
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
