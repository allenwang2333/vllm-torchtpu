# SPDX-License-Identifier: Apache-2.0
"""Full-GSM8K accuracy client for a running DSv4.1 server, plus a paired comparison.

`client` sends every test question once through /v1/chat/completions at a fixed
concurrency: a few-shot multi-turn conversation (seeded choice of train examples),
greedy, thinking disabled, with the request seed fixed. Before the first question
it renders question 0 through the server's own /tokenize and refuses unless the
prompt ends in the chat-mode generation prompt, because a chat-template kwarg that
the template does not read silently leaves thinking on. Each answer is scored by
the GSM8K "#### N" marker when present, else the last number in the response.
Request errors and length truncations are counted, never retried.

`compare` pairs two result files question by question and reports the exact
McNemar test on the discordant pairs.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path

# The DSv4.1 chat template reads `enable_thinking` (and `thinking_mode`);
# vLLM's DeepSeek-V3.2-style tokenizer reads `thinking` or `enable_thinking`.
CHAT_TEMPLATE_KWARGS = {"thinking": False, "enable_thinking": False}
# What a chat-mode (non-thinking) generation prompt ends with.
CHAT_MODE_SUFFIX = "<\uff5cAssistant\uff5c></think>"

_MARKED = re.compile(r"####\s*\$?\s*(-?[\d,]*\.?\d+)")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parse_number(text: str) -> float | None:
    """The answer a response gives: the '#### N' marker, else the last number."""
    marked = _MARKED.findall(text)
    found = marked or _NUMBER.findall(text)
    if not found:
        return None
    try:
        return float(found[-1].replace(",", ""))
    except ValueError:
        return None


def label_of(answer: str) -> float:
    value = parse_number(answer)
    if value is None or "####" not in answer:
        raise ValueError(f"GSM8K reference answer has no '#### N' line: {answer!r}")
    return value


def build_conversations(train, test, *, num_shots: int, shot_seed: int):
    """(messages per test question, the train indices used as shots)."""
    shots = random.Random(shot_seed).sample(range(len(train)), num_shots)
    prefix = []
    for i in shots:
        prefix += [
            {"role": "user", "content": f"Question: {train[i]['question']}\nAnswer:"},
            {"role": "assistant", "content": train[i]["answer"]},
        ]
    conversations = [
        prefix + [{"role": "user", "content": f"Question: {q['question']}\nAnswer:"}]
        for q in test
    ]
    return conversations, shots


def mcnemar_exact(regressions: int, improvements: int) -> float:
    """Two-sided exact McNemar p: binomial(n, 1/2) on the discordant pairs."""
    n = regressions + improvements
    if n == 0:
        return 1.0
    k = min(regressions, improvements)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


async def _post(session, url, payload, timeout):
    import aiohttp

    async with session.post(
        url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)
    ) as response:
        body = await response.text()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {body[:300]}")
        return json.loads(body)


async def render_prompt(session, base, model, messages) -> str:
    tokens = await _post(
        session,
        f"{base}/tokenize",
        {
            "model": model,
            "messages": messages,
            "add_generation_prompt": True,
            "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
        },
        120,
    )
    text = await _post(
        session,
        f"{base}/detokenize",
        {"model": model, "tokens": tokens["tokens"]},
        120,
    )
    return text["prompt"]


async def sample_running(session, base, stop: asyncio.Event, peaks: dict):
    """Track the peak of vllm:num_requests_running, summed and per engine."""
    import aiohttp

    pattern = re.compile(r"^vllm:num_requests_running\{([^}]*)\}\s+([0-9.eE+-]+)$")
    while not stop.is_set():
        try:
            async with session.get(
                f"{base}/metrics", timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                text = await response.text()
            per_engine = {}
            for line in text.splitlines():
                m = pattern.match(line)
                if m:
                    engine = re.search(r'engine="([^"]*)"', m.group(1))
                    key = engine.group(1) if engine else m.group(1)
                    per_engine[key] = per_engine.get(key, 0.0) + float(m.group(2))
            total = sum(per_engine.values())
            peaks["global"] = max(peaks.get("global", 0.0), total)
            for key, value in per_engine.items():
                peaks.setdefault("per_engine", {})
                peaks["per_engine"][key] = max(peaks["per_engine"].get(key, 0.0), value)
        except Exception as exc:  # the sampler is a diagnostic; record, keep going
            peaks["sampler_errors"] = peaks.get("sampler_errors", 0) + 1
            peaks["last_sampler_error"] = repr(exc)[:200]
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=2.0)


async def run_client(args) -> dict:
    import aiohttp

    data = Path(args.data)
    train, test = read_jsonl(data / "train.jsonl"), read_jsonl(data / "test.jsonl")
    if args.limit:
        test = test[: args.limit]
    labels = [label_of(q["answer"]) for q in test]
    conversations, shots = build_conversations(
        train, test, num_shots=args.num_shots, shot_seed=args.seed
    )
    base = args.url.rstrip("/")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        rendered = await render_prompt(session, base, args.model, conversations[0])
        if not rendered.endswith(CHAT_MODE_SUFFIX):
            raise RuntimeError(
                "the server renders question 0 without the chat-mode generation "
                f"prompt (thinking would be on); prompt tail: {rendered[-200:]!r}"
            )
        gate = asyncio.Semaphore(args.concurrency)
        results: list[dict | None] = [None] * len(test)

        async def one(i: int):
            payload = {
                "model": args.model,
                "messages": conversations[i],
                "temperature": 0.0,
                "max_tokens": args.max_tokens,
                "seed": args.seed,
                "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
            }
            record = {"index": i, "label": labels[i]}
            async with gate:
                t0 = time.monotonic()
                try:
                    response = await _post(
                        session, f"{base}/v1/chat/completions", payload, args.timeout
                    )
                    choice = response["choices"][0]
                    text = choice["message"].get("content") or ""
                    pred = parse_number(text)
                    record.update(
                        response=text,
                        reasoning=choice["message"].get("reasoning_content"),
                        finish_reason=choice.get("finish_reason"),
                        completion_tokens=response.get("usage", {}).get(
                            "completion_tokens"
                        ),
                        prompt_tokens=response.get("usage", {}).get("prompt_tokens"),
                        pred=pred,
                        correct=pred is not None and abs(pred - labels[i]) < 1e-6,
                        error=None,
                    )
                except Exception as exc:
                    record.update(correct=False, pred=None, error=repr(exc)[:500])
                record["seconds"] = time.monotonic() - t0
            results[i] = record

        stop = asyncio.Event()
        peaks: dict = {}
        sampler = asyncio.create_task(sample_running(session, base, stop, peaks))
        started = time.monotonic()
        await asyncio.gather(*(one(i) for i in range(len(test))))
        elapsed = time.monotonic() - started
        stop.set()
        await sampler

    with (out / "results.jsonl").open("w") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")
    correct = sum(r["correct"] for r in results)
    summary = {
        "questions": len(results),
        "correct": correct,
        "accuracy": correct / len(results),
        "request_errors": sum(r["error"] is not None for r in results),
        "length_truncations": sum(r.get("finish_reason") == "length" for r in results),
        "unparsed": sum(r["error"] is None and r["pred"] is None for r in results),
        "with_reasoning": sum(bool(r.get("reasoning")) for r in results),
        "completion_tokens": sum(r.get("completion_tokens") or 0 for r in results),
        "elapsed_s": elapsed,
        "running_peaks": peaks,
        "protocol": {
            "endpoint": "/v1/chat/completions",
            "model": args.model,
            "num_shots": args.num_shots,
            "shot_train_indices": shots,
            "seed": args.seed,
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
            "concurrency": args.concurrency,
            "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
            "scoring": "'#### N' marker, else the last number; float equality",
            "train_sha256": sha256(data / "train.jsonl"),
            "test_sha256": sha256(data / "test.jsonl"),
            "limit": args.limit,
        },
        "rendered_prompt_q0": rendered,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def compare(a_dir: Path, b_dir: Path) -> dict:
    """Paired comparison of B against A (A = control)."""
    a = {r["index"]: r for r in read_jsonl(Path(a_dir) / "results.jsonl")}
    b = {r["index"]: r for r in read_jsonl(Path(b_dir) / "results.jsonl")}
    if a.keys() != b.keys():
        raise ValueError("the two runs did not answer the same questions")
    regressions = [i for i in a if a[i]["correct"] and not b[i]["correct"]]
    improvements = [i for i in a if b[i]["correct"] and not a[i]["correct"]]
    n = len(a)
    ca, cb = (
        sum(r["correct"] for r in a.values()),
        sum(r["correct"] for r in b.values()),
    )
    return {
        "questions": n,
        "control_correct": ca,
        "candidate_correct": cb,
        "control_accuracy": ca / n,
        "candidate_accuracy": cb / n,
        "delta_points": 100 * (cb - ca) / n,
        "regressions": len(regressions),
        "improvements": len(improvements),
        "regression_indices": regressions,
        "improvement_indices": improvements,
        "identical_responses": sum(
            a[i].get("response") == b[i].get("response") for i in a
        ),
        "mcnemar_exact_p": mcnemar_exact(len(regressions), len(improvements)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    client = sub.add_parser("client")
    client.add_argument(
        "--data", required=True, help="dir with train.jsonl, test.jsonl"
    )
    client.add_argument("--out", required=True)
    client.add_argument("--url", default="http://127.0.0.1:8000")
    client.add_argument("--model", default="deepseek-v41")
    client.add_argument("--concurrency", type=int, default=384)
    client.add_argument("--num-shots", type=int, default=4)
    client.add_argument("--seed", type=int, default=42)
    client.add_argument("--max-tokens", type=int, default=2048)
    client.add_argument("--timeout", type=float, default=3600)
    client.add_argument("--limit", type=int, default=0, help="first N questions")
    cmp = sub.add_parser("compare")
    cmp.add_argument("control")
    cmp.add_argument("candidate")
    args = parser.parse_args()
    if args.command == "client":
        summary = asyncio.run(run_client(args))
        print(
            json.dumps({k: v for k, v in summary.items() if k != "rendered_prompt_q0"})
        )
    else:
        print(json.dumps(compare(args.control, args.candidate), indent=2))


if __name__ == "__main__":
    main()
