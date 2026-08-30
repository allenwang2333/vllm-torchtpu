#!/usr/bin/env python3
"""GSM8K accuracy + degeneration probe against a vLLM chat endpoint.

Written for the hierarchical-EP correctness investigation (see
K3_HIERARCHICAL_EP_PLAN.md, "CORRECTNESS FAILURE at TP32"). The TP32 failure did
NOT present primarily as an accuracy drop -- accuracy fell only 4pp -- it
presented as the model solving the problem and then failing to terminate:

    47% of responses degenerated into '..enterprise..enterprise..' loops,
    every response hit the token cap, and the tails carried unrelated CJK spam.

So an accuracy number alone is the wrong instrument: at 100 documents the
binomial error near p=0.9 is ~3pp and a 4pp shift is invisible, while the
degeneration rate moved from 0% to 47% and is unmissable. This script reports
both, and the degeneration columns are the ones to read first.

Unlike scripts/eval_kimi_k3_gsm8k.py this talks to /v1/chat/completions and
lets the server apply the model's own chat template, so it works for any
instruct model rather than only K3's XTML prompt format.

Usage:
    python gsm8k_degeneration_probe.py --base-url http://127.0.0.1:8010/v1 \
        --model moonshotai/Kimi-Linear-48B-A3B-Instruct --limit 100 \
        --output arm.jsonl --summary arm.json
    python gsm8k_degeneration_probe.py --canary ...   # one-line reproducer
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import re
import statistics
import sys
import threading
import time
from pathlib import Path

import requests

NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
CJK_RE = re.compile("[\u3000-\u9fff\uff00-\uffef]")

INSTRUCTION = ("Solve this math problem. Show concise work and end with "
               "'#### ' followed by only the numeric answer.\n\n{question}")

# The single-line prompt that reproduced the TP32 failure in seconds. A healthy
# model answers in a few dozen tokens; the broken one ran to the cap.
CANARY_PROMPT = "What is 7 plus 5?"


def normalize_number(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.replace(",", "").rstrip(".")
    try:
        number = float(value)
    except ValueError:
        return value
    return str(int(number)) if number.is_integer() else str(number)


def extract_gold(answer: str) -> str | None:
    match = re.search(r"####\s*([^\n]+)", answer)
    numbers = NUMBER_RE.findall(match.group(1)) if match else []
    return normalize_number(numbers[-1] if numbers else None)


def extract_prediction(text: str) -> str | None:
    marked = re.findall(r"####\s*(-?\d[\d,]*(?:\.\d+)?)", text)
    numbers = marked or NUMBER_RE.findall(text)
    return normalize_number(numbers[-1] if numbers else None)


def max_ngram_repeat(text: str, n: int = 4) -> int:
    """How many times the most frequent word n-gram occurs.

    A correct chain-of-thought reuses phrases a handful of times at most; a
    repetition loop drives one n-gram into the tens or hundreds.
    """
    words = text.split()
    if len(words) < n * 3:
        return 0
    counts = collections.Counter(
        tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    return max(counts.values())


def tail_chunk_repeat(text: str, chunk: int = 30) -> int:
    """Occurrences of the final `chunk` characters elsewhere in the text.

    Catches loops that word-splitting misses -- the observed failure emitted
    '..enterprise. final..enterprise. final..' with no spaces in places.
    """
    stripped = text.strip()
    if len(stripped) < chunk * 3:
        return 0
    return stripped.count(stripped[-chunk:])


def cjk_fraction(text: str) -> float:
    tail = text[-400:]
    if not tail:
        return 0.0
    return len(CJK_RE.findall(tail)) / len(tail)


def classify(text: str) -> dict:
    ngram = max_ngram_repeat(text)
    tail = tail_chunk_repeat(text)
    cjk = cjk_fraction(text)
    return {
        "max_ngram_repeat": ngram,
        "tail_chunk_repeat": tail,
        "cjk_tail_fraction": round(cjk, 4),
        # Thresholds are deliberately loose. Both arms are scored by the same
        # rule, so the comparison holds even if the absolute rate is off by a
        # few percent; the signal being looked for is 0% versus ~50%.
        "degenerate": bool(ngram >= 5 or tail >= 4 or cjk > 0.05),
    }


def summarize(records: list[dict], label: str, model: str,
              requested: int) -> dict:
    ok = [r for r in records if r.get("ok")]
    tokens = [r["completion_tokens"] for r in ok if r.get("completion_tokens")]
    finish = collections.Counter(r.get("finish_reason") for r in ok)
    correct = sum(1 for r in ok if r.get("correct"))
    degenerate = sum(1 for r in ok if r.get("degenerate"))
    return {
        "label":
        label,
        "model":
        model,
        "requested":
        requested,
        "completed":
        len(ok),
        "failed":
        len(records) - len(ok),
        "correct":
        correct,
        "accuracy":
        correct / len(ok) if ok else None,
        "degenerate":
        degenerate,
        "degeneration_rate":
        degenerate / len(ok) if ok else None,
        "cjk_tail":
        sum(1 for r in ok if r.get("cjk_tail_fraction", 0) > 0.05),
        "finish_reason":
        dict(finish),
        "finish_length_rate":
        (finish.get("length", 0) / len(ok) if ok else None),
        "mean_completion_tokens":
        (statistics.mean(tokens) if tokens else None),
        "median_completion_tokens":
        (statistics.median(tokens) if tokens else None),
    }


def _score_record(raw: dict) -> dict:
    """Normalise one record from whichever harness produced it.

    Two formats are accepted, because the two harnesses in use here disagree:

    * ``eval_kimi_k3_gsm8k.py`` -- completion in ``content``, success signalled
      by ``status_code == 200``.
    * ``lm_eval --log_samples`` (what ``scripts/k3/eval_gsm8k.sh`` runs, and so
      what teammates' numbers come from) -- completion in ``filtered_resps`` or
      ``resps``, correctness already scored into ``exact_match``.

    lm-eval carries no ``finish_reason`` or token counts, so those columns come
    out empty on that path. The degeneration columns -- the ones that separated
    job 482 from a healthy run -- do not depend on them.
    """
    record: dict = {"index": raw.get("doc_id", raw.get("index"))}

    if "filtered_resps" in raw or "resps" in raw:
        responses = raw.get("filtered_resps") or raw.get("resps") or []
        # resps nests one list per repeat; filtered_resps is already flat.
        while isinstance(responses, list) and responses and isinstance(
                responses[0], list):
            responses = responses[0]
        text = responses[0] if responses else ""
        text = text if isinstance(text, str) else str(text)
        record["ok"] = bool(text)
        record["content"] = text
        for key in ("exact_match", "acc", "em"):
            if key in raw:
                record["correct"] = bool(float(raw[key]))
                break
        else:
            record["correct"] = None
    else:
        text = raw.get("content") or ""
        # Three producers, three success conventions: this script writes an
        # "ok" flag, eval_kimi_k3_gsm8k.py writes status_code, and a partial
        # file from either may have neither on a failed row.
        if "ok" in raw:
            record["ok"] = bool(raw["ok"])
        else:
            record["ok"] = (raw.get("status_code") == 200
                            and "error" not in raw)
        record["content"] = text
        record["correct"] = raw.get("correct")
        record["finish_reason"] = raw.get("finish_reason")
        usage = raw.get("usage") or {}
        record["completion_tokens"] = (raw.get("completion_tokens")
                                       or usage.get("completion_tokens"))
        if record["ok"] and record["correct"] is None:
            record["correct"] = (extract_prediction(text) == raw.get("gold"))

    if record["ok"]:
        record.update(classify(text))
    return record


def run_score(args) -> int:
    """Score a jsonl another harness produced.

    Both harnesses write the model's completion per document but neither asks
    whether it degenerated. Re-scoring their output here adds exactly that,
    without changing the accuracy number they report -- so the accuracy stays
    comparable to whatever those runs are being compared against.
    """
    records = []
    seen_filters: set[str] = set()
    skipped = 0
    with args.score.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if "filter" in raw:
                seen_filters.add(raw["filter"])
                if raw["filter"] != args.filter:
                    skipped += 1
                    continue
            records.append(_score_record(raw))

    # If the requested filter matched nothing, scoring silently on zero records
    # would report accuracy=None and look like an empty run rather than a wrong
    # flag. Fall back to whichever filter is present and say so.
    if not records and seen_filters:
        fallback = sorted(seen_filters)[0]
        print(f"note: no records with filter={args.filter!r}; using "
              f"{fallback!r} (available: {sorted(seen_filters)})")
        with args.score.open(encoding="utf-8") as handle:
            records = [
                _score_record(json.loads(line)) for line in handle
                if line.strip() and json.loads(line).get("filter") == fallback
            ]
        skipped = 0
    if skipped:
        print(f"filter={args.filter!r}: kept {len(records)}, skipped "
              f"{skipped} from {sorted(seen_filters - {args.filter})}")

    summary = summarize(records, args.label, args.model, len(records))
    summary["scored_from"] = str(args.score)
    print(json.dumps(summary, indent=2))
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2))
    return 0


def chat(base_url: str,
         model: str,
         prompt: str,
         max_tokens: int,
         timeout: float,
         completions: bool = False) -> dict:
    """POST one prompt, via chat or raw completions.

    ``completions=True`` renders K3's XTML prompt directly and hits
    /v1/completions. Needed because the full-K3 TP32 launcher configures no
    chat template -- its packaged one is the multimodal variant -- so
    /v1/chat/completions is not usable there.
    """
    if completions:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from kimi_k3_prompt import render_prompt
        response = requests.post(
            f"{base_url.rstrip('/')}/completions",
            json={
                "model": model,
                "prompt": render_prompt(prompt),
                "temperature": 0,
                "max_tokens": max_tokens,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        body = response.json()
        # Normalise to the chat shape so callers do not branch.
        for choice in body.get("choices", []):
            choice["message"] = {"content": choice.get("text") or ""}
        return body

    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json={
            "model": model,
            "messages": [{
                "role": "user",
                "content": prompt
            }],
            "temperature": 0,
            "max_tokens": max_tokens,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def run_canary(args) -> int:
    body = chat(args.base_url,
                args.model,
                CANARY_PROMPT,
                args.max_tokens,
                args.timeout,
                completions=args.completions)
    choice = body["choices"][0]
    text = choice.get("message", {}).get("content") or ""
    verdict = classify(text)
    tokens = (body.get("usage") or {}).get("completion_tokens")
    print(f"   prompt:        {CANARY_PROMPT!r}")
    print(f"   finish_reason: {choice.get('finish_reason')}")
    print(f"   tokens:        {tokens}")
    print(f"   degenerate:    {verdict['degenerate']}  "
          f"(ngram={verdict['max_ngram_repeat']}, "
          f"tail={verdict['tail_chunk_repeat']}, "
          f"cjk={verdict['cjk_tail_fraction']})")
    print(f"   text: {text[:300]!r}")
    if len(text) > 300:
        print(f"   tail: {text[-200:]!r}")
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(
                {
                    "canary": True,
                    "finish_reason": choice.get("finish_reason"),
                    "completion_tokens": tokens,
                    "text": text,
                    **verdict,
                },
                indent=2,
                ensure_ascii=False))
    # A degenerate canary is the whole point of the run, not an error: exit 0
    # and let the caller compare arms.
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8010/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--canary",
                        action="store_true",
                        help="run only the one-line reproducer prompt")
    parser.add_argument("--completions",
                        action="store_true",
                        help="use /v1/completions with K3's XTML prompt "
                        "instead of /v1/chat/completions; required when "
                        "the server has no text chat template")
    parser.add_argument("--score",
                        type=Path,
                        default=None,
                        help="score an existing jsonl instead of querying a "
                        "server; records need 'content' and 'correct' or "
                        "'gold'. Lets eval_kimi_k3_gsm8k.py -- the harness "
                        "jobs 460 and 482 used, so the accuracy stays "
                        "comparable to theirs -- also yield degeneration "
                        "numbers.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--label", default="")
    parser.add_argument("--filter",
                        default="flexible-extract",
                        help="lm-eval writes one record per (doc, filter) "
                        "pair, so a 1319-doc gsm8k run yields 2638 lines "
                        "and scoring them all double-counts every "
                        "document. Keep only this filter's records.")
    args = parser.parse_args()

    if args.score:
        return run_score(args)
    if args.canary:
        return run_canary(args)

    import pandas as pd
    from huggingface_hub import hf_hub_download

    dataset_path = hf_hub_download("openai/gsm8k",
                                   "main/test-00000-of-00001.parquet",
                                   repo_type="dataset")
    rows = pd.read_parquet(dataset_path).iloc[args.offset:args.offset +
                                              args.limit]

    lock = threading.Lock()
    records: list[dict] = []
    started = time.monotonic()

    def evaluate(item) -> dict:
        index, row = item
        record = {
            "index": int(index),
            "gold": extract_gold(row["answer"]),
        }
        request_started = time.monotonic()
        try:
            body = chat(args.base_url,
                        args.model,
                        INSTRUCTION.format(question=row["question"]),
                        args.max_tokens,
                        args.timeout,
                        completions=args.completions)
            choice = body["choices"][0]
            text = choice.get("message", {}).get("content") or ""
            record.update({
                "content":
                text,
                "finish_reason":
                choice.get("finish_reason"),
                "completion_tokens": (body.get("usage")
                                      or {}).get("completion_tokens"),
                "prediction":
                extract_prediction(text),
                **classify(text),
            })
            record["correct"] = record["prediction"] == record["gold"]
            record["ok"] = True
        except Exception as error:  # noqa: BLE001 - report, do not abort
            record["error"] = repr(error)
            record["ok"] = False
        record["latency_seconds"] = time.monotonic() - request_started
        return record

    handle = args.output.open("w", encoding="utf-8") if args.output else None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.concurrency) as pool:
            futures = [pool.submit(evaluate, item) for item in rows.iterrows()]
            for future in concurrent.futures.as_completed(futures):
                record = future.result()
                with lock:
                    records.append(record)
                    if handle:
                        handle.write(
                            json.dumps(record, ensure_ascii=False) + "\n")
                        handle.flush()
                    done = len(records)
                    ok = [r for r in records if r.get("ok")]
                    correct = sum(1 for r in ok if r.get("correct"))
                    degen = sum(1 for r in ok if r.get("degenerate"))
                    print(
                        f"  {done}/{len(rows)} acc="
                        f"{correct / len(ok) if ok else 0:.3f} degen="
                        f"{degen / len(ok) if ok else 0:.3f}",
                        flush=True)
    finally:
        if handle:
            handle.close()

    summary = summarize(records, args.label, args.model, len(rows))
    summary["elapsed_seconds"] = time.monotonic() - started
    print(json.dumps(summary, indent=2))
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
