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
import ast
import base64
import gc
import io
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image


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
        default=0,
        help="Number of samples to evaluate (0 for all dataset samples)",
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
        "--config",
        default="standard (4 options)",
        help="Dataset config/subset name (default: 'standard (4 options)')",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent parallel HTTP requests to send (default: 1)",
    )
    return parser.parse_args()


def encode_image_base64(image) -> str:
    """Convert PIL Image to JPEG base64 string."""
    buffered = io.BytesIO()
    image.convert("RGB").save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def extract_answer(response_text: str,
                   options: list[str] | None = None) -> str:
    """Extract choice letter (A-J) from generated response text."""
    if not response_text:
        return "UNKNOWN"

    clean_text = response_text.replace("**", "").replace("__", "").strip()

    # 1. Check if the entire response or the first/last non-empty line is just a choice letter
    lines = [line.strip() for line in clean_text.splitlines() if line.strip()]
    if lines:
        for line in [lines[-1], lines[0]]:
            direct_line_match = re.match(
                r"^(?:(?:the\s+)?(?:correct\s+)?(?:answer|option|choice|solution)\s*(?:is|=|:)\s*)?\(?([A-J])\)?[\.\:\s]*$",
                line,
                re.IGNORECASE,
            )
            if direct_line_match:
                return direct_line_match.group(1).upper()

    # 2. Strict conclusion regexes (searched across entire text, taking the LAST match)
    conclusion_patterns = [
        r"(?:therefore|thus|hence|so|consequently|in summary|in conclusion),?\s*(?:the)?\s*(?:correct)?\s*(?:answer|option|choice|solution)\s*(?:is|=|:)?\s*\(?([A-J])\)?\b",
        r"\b(?:final\s+)?(?:answer|option|choice|solution)\s*(?:is|=|:)\s*\(?([A-J])\)?\b",
        r"\b(?:the\s+)?correct\s+(?:answer|option|choice|solution)\s*(?:is|=|:)?\s*\(?([A-J])\)?\b",
        r"\boption\s*\(?([A-J])\)?\s*is\s*(?:the\s+)?(?:correct|right)\b",
        r"\\boxed\{([A-J])\}",
        r"\(?([A-J])\)\s*is\s*(?:the\s+)?(?:correct|right)",
    ]
    for pattern in conclusion_patterns:
        matches = list(re.finditer(pattern, clean_text, re.IGNORECASE))
        if matches:
            return matches[-1].group(1).upper()

    # 3. Look at the last 300 characters: find the LAST explicitly formatted option like (A) or Option A
    tail_text = clean_text[-300:]
    enclosed_matches = list(
        re.finditer(r"(?:\boption\s+|\()([A-J])(?:\)|\.|\b)", tail_text,
                    re.IGNORECASE))
    if enclosed_matches:
        return enclosed_matches[-1].group(1).upper()

    # 4. Fallback to candidate option text matching against options
    if options and isinstance(options, list):
        clean_resp = clean_text.lower()
        for idx, opt in enumerate(options):
            if opt and str(opt).strip().lower() in clean_resp:
                return chr(65 + idx)

    # 5. Last standalone letter in the tail
    letter_matches = list(re.finditer(r"\b([A-D])\b", tail_text))
    if letter_matches:
        return letter_matches[-1].group(1).upper()

    return "UNKNOWN"


def load_mmmu_samples(
        dataset_name: str = "MMMU/MMMU",
        limit: int = 100,
        allow_synthetic: bool = False,
        config_name: str = "standard (4 options)"
) -> tuple[list[dict], str, bool]:
    """Load MMMU or MMMU-Pro validation samples from HuggingFace datasets or fallback dataset."""
    try:
        from datasets import load_dataset

        print(
            f"Loading {dataset_name} ({config_name}) dataset from HuggingFace (limit={limit})..."
        )
        samples = []
        if "MMMU_Pro" in dataset_name or "mmmu_pro" in dataset_name.lower():
            # MMMU-Pro dataset loading
            target_ds = "MMMU/MMMU_Pro"
            if config_name and config_name != "all":
                configs = [config_name]
            else:
                configs = [
                    "standard (4 options)", "standard (10 options)", "vision"
                ]
            for config in configs:
                if limit > 0 and len(samples) >= limit:
                    break
                ds = None
                for split_name in ["test", "validation"]:
                    try:
                        loaded_ds = load_dataset(target_ds,
                                                 config,
                                                 split=split_name)
                        if loaded_ds:
                            ds = loaded_ds
                            break
                    except Exception:  # noqa: BLE001
                        continue
                if ds is None:
                    continue
                for item in ds:
                    if limit > 0 and len(samples) >= limit:
                        break
                    images = []
                    for k in sorted(item.keys()):
                        val = item[k]
                        if val is not None and (isinstance(val, Image.Image)
                                                or hasattr(val, "convert")):
                            images.append(val)
                    if not images:
                        continue
                    question = item.get("question", "")
                    options_raw = item.get("options", [])
                    if isinstance(options_raw, str):
                        try:
                            options = ast.literal_eval(options_raw)
                        except Exception:  # noqa: BLE001
                            options = []
                    elif isinstance(options_raw, list):
                        options = options_raw
                    else:
                        options = []

                    answer = str(item.get("answer", "")).strip().upper()
                    if not answer or answer in ("NONE", "UNKNOWN"):
                        continue
                    samples.append({
                        "id":
                        item.get("id", f"mmmu_pro_{len(samples)}"),
                        "question":
                        question,
                        "options":
                        options,
                        "answer":
                        answer,
                        "images":
                        images,
                        "image":
                        images[0],
                    })

            # If MMMU_Pro has no labeled split with answers, fall back to MMMU/MMMU validation split
            if not samples:
                print(
                    "MMMU_Pro contains no labeled samples with answers; falling back to MMMU/MMMU validation split..."
                )
                return load_mmmu_samples("MMMU/MMMU",
                                         limit=limit,
                                         allow_synthetic=allow_synthetic,
                                         config_name=config_name)
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
                except Exception:  # noqa: BLE001, S112
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
                            options = ast.literal_eval(options_raw)
                        except Exception:  # noqa: BLE001
                            options = []
                    elif isinstance(options_raw, list):
                        options = options_raw
                    else:
                        options = []

                    answer = str(item.get("answer", "")).strip().upper()
                    if not answer or answer in ("NONE", "UNKNOWN"):
                        continue

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
                f"No valid samples with answers loaded from HuggingFace {dataset_name} dataset."
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
        samples = []
        choices = ["A", "B", "C", "D"]
        colors = ["red", "blue", "green", "yellow"]
        for i in range(min(limit, 20)):
            color = choices[i % 4]
            img = Image.new("RGB", (224, 224), color=colors[i % 4])
            samples.append({
                "id":
                f"synthetic_{i}",
                "question":
                ("What is the dominant background color of this image?\n(A)"
                 " Red\n(B) Blue\n(C) Green\n(D) Yellow"),
                "options": ["Red", "Blue", "Green", "Yellow"],
                "answer":
                color,
                "image":
                img,
            })
        return samples, "synthetic_color_shapes", True


def evaluate_sample(sample: dict, host: str, port: int,
                    model: str) -> tuple[bool, str, str]:
    """Send a single multimodal request to vLLM /v1/chat/completions."""
    url = f"http://{host}:{port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer EMPTY",
    }
    question_clean = re.sub(r"<image \d+>", "", sample["question"]).strip()
    prompt_text = question_clean

    options = sample.get("options", [])
    if (isinstance(options, list) and options and "(A)" not in prompt_text
            and "A." not in prompt_text):
        prompt_text += "\n\nOptions:\n" + "\n".join(
            f"({chr(65+idx)}) {opt}" for idx, opt in enumerate(options))

    prompt_text += (
        "\n\nPlease think step-by-step and output your final choice at the end in the format: 'Answer: (X)' where X is the correct option letter."
    )

    content = []
    images = sample.get("images") or [sample["image"]]
    for img in images:
        img_b64 = encode_image_base64(img)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_b64}"
            },
        })
    content.append({
        "type": "text",
        "text": prompt_text,
    })

    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": content,
        }],
        "max_tokens": 512,
        "temperature": 0.0,
    }

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url,
                                 headers=headers,
                                 json=payload,
                                 timeout=600)
            if resp.status_code == 200:
                res_json = resp.json()
                generated_text = res_json["choices"][0]["message"]["content"]
                predicted = extract_answer(generated_text,
                                           options=sample.get("options"))
                is_correct = predicted == sample["answer"]
                print(
                    f"Sample {sample.get('id')}: Pred={predicted}, Truth={sample['answer']}, Match={is_correct}"
                )
                gc.collect()
                return is_correct, predicted, sample["answer"]
            print(
                f"[ERROR-HTTP-{resp.status_code}] Sample {sample.get('id', 'unknown')} attempt {attempt}: {resp.text[:200]}"
            )
            if attempt == max_retries:
                raise RuntimeError(
                    f"HTTP Error {resp.status_code} for sample {sample.get('id', 'unknown')}: {resp.text[:500]}"
                )
        except Exception as e:
            print(
                f"[ERROR-EXC] Sample {sample.get('id', 'unknown')} attempt {attempt}: {e}"
            )
            if attempt == max_retries:
                raise RuntimeError(
                    f"Request failure for sample {sample.get('id', 'unknown')}: {e}"
                ) from e
        time.sleep(2 * attempt)

    gc.collect()
    raise RuntimeError(
        f"Evaluation failed for sample {sample.get('id', 'unknown')} after {max_retries} attempts."
    )


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    print("====================================================")
    print("  MMMU / MMMU-Pro Generative Vision Evaluation")
    print(f"  Target: http://{args.host}:{args.port}/v1/chat/completions")
    print(f"  Concurrency: {args.concurrency} parallel workers")
    print("====================================================")

    samples, dataset_source, is_synthetic = load_mmmu_samples(
        args.dataset, args.limit, args.allow_synthetic, args.config)
    if not samples:
        print("ERROR: No samples loaded for MMMU evaluation.")
        sys.exit(1)

    correct_count = 0
    total_count = len(samples)
    results = []
    completed_idx = 0
    lock = threading.Lock()

    t0 = time.time()

    def eval_worker(sample_tuple):
        seq_id, sample = sample_tuple
        is_correct, pred, target = evaluate_sample(sample, args.host,
                                                   args.port, args.model)
        return seq_id, sample["id"], is_correct, pred, target

    indexed_samples = list(enumerate(samples, 1))

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = [
            executor.submit(eval_worker, item) for item in indexed_samples
        ]
        for future in as_completed(futures):
            _seq_id, sample_id, is_correct, pred, target = future.result()
            with lock:
                completed_idx += 1
                if is_correct:
                    correct_count += 1
                results.append({
                    "id": sample_id,
                    "predicted": pred,
                    "ground_truth": target,
                    "is_correct": is_correct,
                })

                if completed_idx % 10 == 0 or completed_idx == total_count:
                    acc = correct_count / completed_idx * 100.0
                    print(
                        f"[{completed_idx}/{total_count}] Current Accuracy: {acc:.2f}% ({correct_count}/{completed_idx})"
                    )
                    out_path = os.path.join(args.results_dir,
                                            "mmmu_eval_results.json")
                    with open(out_path, "w") as f:
                        json.dump(
                            {
                                "model": args.model,
                                "dataset_source": dataset_source,
                                "is_synthetic": is_synthetic,
                                "samples": completed_idx,
                                "accuracy": acc,
                                "correct": correct_count,
                                "details": results,
                            },
                            f,
                            indent=2)
                        f.flush()
                        os.fsync(f.fileno())

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
        f.flush()
        os.fsync(f.fileno())
    print(f"Results saved to: {output_file}")
    if total_count > 0 and correct_count == 0:
        raise RuntimeError(
            f"MMMU Evaluation FAILED: 0/{total_count} correct predictions. Check server logs/health."
        )


if __name__ == "__main__":
    main()
