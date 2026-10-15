# SPDX-License-Identifier: Apache-2.0
"""Checks real-weight disaggregated MTP with GSM8K and chunked prefill traffic."""

import argparse
import json
import math
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

COUNTERS = (
    "request_success",
    "spec_decode_num_drafts",
    "spec_decode_num_draft_tokens",
    "spec_decode_num_accepted_tokens",
)
EVAL_LIMIT = 128
BENCHMARK_MAIN = (
    "from vllm.utils.argparse_utils import FlexibleArgumentParser; "
    "from vllm.benchmarks.serve import add_cli_args, main; "
    "p = FlexibleArgumentParser(); add_cli_args(p); main(p.parse_args())"
)


def metric_values(body: str) -> dict[str, float]:
    """Sums Prometheus counter series across workers."""
    values: dict[str, float] = {}
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        name = fields[0].split("{", 1)[0].removeprefix("vllm:").removesuffix("_total")
        if name in COUNTERS:
            value = float(fields[1])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite metric: {line}")
            values[name] = values.get(name, 0.0) + value
    return values


def check_path(before, after, *, mode: str, requests: int) -> dict:
    """Requires completed requests on the intended engines and real MTP activity."""
    prefill = after["prefill"]["request_success"] - before["prefill"]["request_success"]
    delta = {name: after["decode"][name] - before["decode"][name] for name in COUNTERS}
    drafts = delta["spec_decode_num_drafts"]
    proposed = delta["spec_decode_num_draft_tokens"]
    accepted = delta["spec_decode_num_accepted_tokens"]
    result = {
        "mode": mode,
        "prefill_requests": prefill,
        "decode_requests": delta["request_success"],
        "drafts": drafts,
        "proposed_tokens": proposed,
        "accepted_tokens": accepted,
        "acceptance_rate": accepted / proposed if proposed > 0 else 0,
        "acceptance_length": 1 + accepted / drafts if drafts > 0 else 0,
    }
    print("MTP_PATH", json.dumps(result), flush=True)
    if delta["request_success"] < requests:
        raise ValueError(f"Incomplete decode requests: {result}")
    if (mode == "direct" and prefill != 0) or (mode != "direct" and prefill < requests):
        raise ValueError(f"Unexpected prefill routing: {result}")
    if drafts <= 0 or proposed <= 0 or not 0 < accepted <= proposed:
        raise ValueError(f"Invalid MTP counters: {result}")
    if result["acceptance_rate"] < 0.5:
        raise ValueError(f"MTP acceptance below 0.5: {result}")
    return result


def check_accuracy(scores: dict[str, float], limit: int) -> dict:
    """Allows at most two fewer correct answers through P/D than direct decode."""
    passed = (
        all(math.isfinite(s) and 0.70 <= s <= 1 for s in scores.values())
        and (scores["direct"] - scores["pd"]) * limit <= 2.00001
    )
    return {
        "scores": scores,
        "limit": limit,
        "allowed_answer_difference": 2,
        "passed": passed,
    }


def run(command: list[str], log: Path) -> None:
    """Records a standard evaluator or benchmark and propagates its failure."""
    with log.open("w") as stream:
        process = subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT, timeout=7200
        )
    print(log.read_text(), flush=True)
    process.check_returncode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prefill-url", "decode-url", "proxy-url", "tokenizer"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def snapshot(label):
        counters = {}
        for role, url in (("prefill", args.prefill_url), ("decode", args.decode_url)):
            with urllib.request.urlopen(f"{url}/metrics", timeout=30) as response:
                body = response.read().decode()
            (args.output_dir / f"{label}-{role}.metrics").write_text(body)
            counters[role] = metric_values(body)
            required = COUNTERS if role == "decode" else ("request_success",)
            if any(name not in counters[role] for name in required):
                raise ValueError(f"Missing {role} metrics: {required}")
        return counters

    def finish(mode, before, requests):
        # Request counters reach /metrics on the engine's periodic stats update.
        deadline = time.monotonic() + 60
        while True:
            after = snapshot(f"{mode}-after")
            done = (
                after["decode"]["request_success"] - before["decode"]["request_success"]
            )
            prefilled = (
                after["prefill"]["request_success"]
                - before["prefill"]["request_success"]
            )
            if (
                done >= requests and (mode == "direct" or prefilled >= requests)
            ) or time.monotonic() >= deadline:
                return check_path(before, after, mode=mode, requests=requests)
            time.sleep(2)

    evidence = []
    scores = {}
    for mode, url in (("pd", args.proxy_url), ("direct", args.decode_url)):
        before = snapshot(f"{mode}-before")
        run(
            [
                sys.executable,
                "-m",
                "lm_eval",
                "--model",
                "local-completions",
                "--model_args",
                f"model=qwen38-mtp,base_url={url}/v1/completions,"
                f"tokenizer={args.tokenizer},tokenizer_backend=huggingface,"
                "tokenized_requests=False,num_concurrent=8,max_retries=1,"
                "max_gen_toks=2048,max_length=40960,timeout=1800",
                "--tasks",
                "gsm8k",
                "--num_fewshot",
                "32",
                "--limit",
                str(EVAL_LIMIT),
                "--seed",
                "1234,1234,1234,1234",
                "--log_samples",
                "--output_path",
                str(args.output_dir / f"eval-{mode}"),
            ],
            args.output_dir / f"eval-{mode}.log",
        )
        evidence.append(finish(mode, before, EVAL_LIMIT))
        results = list((args.output_dir / f"eval-{mode}").rglob("results_*.json"))
        if len(results) != 1:
            raise ValueError(f"Expected one GSM8K result: {results}")
        scores[mode] = json.loads(results[0].read_text())["results"]["gsm8k"][
            "exact_match,flexible-extract"
        ]
    accuracy = check_accuracy(scores, EVAL_LIMIT)
    (args.output_dir / "accuracy.json").write_text(json.dumps(accuracy, indent=2))
    print("ACCURACY", json.dumps(accuracy), flush=True)
    if not accuracy["passed"]:
        raise ValueError("GSM8K accuracy gate failed")

    before = snapshot("chunked-before")
    # Each prompt exceeds the producer's 16k chunk budget and crosses KV blocks.
    run(
        [
            sys.executable,
            "-c",
            BENCHMARK_MAIN,
            "--backend",
            "vllm",
            "--base-url",
            args.proxy_url,
            "--model",
            "qwen38-mtp",
            "--tokenizer",
            args.tokenizer,
            "--dataset-name",
            "random",
            "--random-input-len",
            "32768",
            "--random-output-len",
            "256",
            "--random-range-ratio",
            "0.0",
            "--num-prompts",
            "16",
            "--max-concurrency",
            "8",
            "--request-rate",
            "inf",
            "--ignore-eos",
            "--temperature",
            "0",
            "--seed",
            "42",
            "--save-result",
            "--save-detailed",
            "--result-dir",
            str(args.output_dir),
            "--result-filename",
            "chunked.json",
        ],
        args.output_dir / "chunked.log",
    )
    evidence.append(finish("chunked", before, 16))
    benchmark = json.loads((args.output_dir / "chunked.json").read_text())
    if benchmark["completed"] != 16:
        raise ValueError(f"Incomplete chunked requests: {benchmark['completed']}")
    (args.output_dir / "path-evidence.json").write_text(json.dumps(evidence, indent=2))
    print("RESULT: PASS (Qwen3.8 FP8 PP4xPCP8 MTP1 -> TP32 DCP8 MTP3)", flush=True)


if __name__ == "__main__":
    main()
