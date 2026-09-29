# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact-input ABBA streaming measurements and paired 100-question GSM8K checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path

import requests

ARMS = ("baseline", "candidate", "candidate", "baseline")
CELLS = ((1, 1), (1, 4), (8192, 1), (8192, 4))
GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/test.jsonl"
)
PERFORMANCE_SETTINGS = {"output_tokens": 1024, "requests": 10, "warmup_waves": 2}
SAMPLING = {"temperature": 0, "seed": 0, "add_special_tokens": False}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def events(lines):
    data = []
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if not line:
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
    if data:
        yield "\n".join(data)


def consume_stream(lines, *, started, clock=time.perf_counter):
    tokens, arrivals, text = [], [], []
    usage, finish_reason, done = None, None, False
    for event in events(lines):
        if event == "[DONE]":
            done = True
            break
        body = json.loads(event)
        if body.get("error"):
            raise ValueError(f"server error: {body['error']}")
        usage = body.get("usage") or usage
        choices = body.get("choices", [])
        if len(choices) > 1:
            raise ValueError("expected one completion")
        for choice in choices:
            delta = choice.get("token_ids") or []
            if any(type(token) is not int or token < 0 for token in delta):
                raise ValueError("invalid output token IDs")
            if delta:
                arrivals.append({"seconds": clock() - started, "tokens": delta})
                tokens.extend(delta)
            text.append(choice.get("text") or "")
            finish_reason = choice.get("finish_reason") or finish_reason
    if not done or not tokens or usage is None:
        raise ValueError("incomplete stream: require DONE, token IDs and usage")
    if finish_reason not in ("stop", "length"):
        raise ValueError("missing or unexpected completion finish reason")
    if usage.get("completion_tokens") != len(tokens):
        raise ValueError("token IDs disagree with completion usage")
    return {
        "token_ids": tokens,
        "arrivals": arrivals,
        "usage": usage,
        "text": "".join(text),
        "finish_reason": finish_reason,
    }


def request(base_url, model, prompt, output_tokens, *, exact, timeout):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_tokens,
        **SAMPLING,
        "ignore_eos": exact,
        "return_token_ids": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    try:
        with requests.post(
            base_url.rstrip("/") + "/v1/completions",
            json=payload,
            stream=True,
            timeout=(30, timeout),
        ) as response:
            response.raise_for_status()
            row = consume_stream(response.iter_lines(chunk_size=None), started=started)
        ended = time.perf_counter()
        if row["usage"].get("prompt_tokens") != len(prompt):
            raise ValueError("server input usage differs from cached token IDs")
        count = len(row["token_ids"])
        if exact and count != output_tokens:
            raise ValueError(f"expected {output_tokens} output tokens, got {count}")
        if exact and len(row["arrivals"]) < 2:
            raise ValueError("stream was fully buffered; token latency is unobservable")
        first, last = row["arrivals"][0]["seconds"], row["arrivals"][-1]["seconds"]
        row.update(
            started=started,
            ended=ended,
            ttft_ms=1000 * first,
            tpot_ms=1000 * (last - first) / (count - 1) if count > 1 else None,
            e2e_ms=1000 * (ended - started),
            single_token_chunks=all(len(a["tokens"]) == 1 for a in row["arrivals"]),
        )
        return row
    except Exception as error:
        return {"started": started, "ended": time.perf_counter(), "error": repr(error)}


def tokenize(base_url, model, **payload):
    response = requests.post(
        base_url.rstrip("/") + "/tokenize",
        json={"model": model, "add_special_tokens": False, **payload},
        timeout=120,
    )
    response.raise_for_status()
    tokens = response.json()["tokens"]
    if not tokens or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("tokenizer returned invalid token IDs")
    return tokens


def answer(text):
    matches = re.findall(r"####\s*(-?\d[\d,]*(?:\.\d+)?)", text)
    return str(Decimal(matches[-1].replace(",", "")).normalize()) if matches else None


def prepare(args):
    if args.output.exists():
        raise ValueError(
            "inputs already exist; reuse the cached file without rewriting"
        )
    if args.dataset:
        raw = args.dataset.read_bytes()
        source = str(args.dataset.resolve())
    else:
        response = requests.get(GSM8K_URL, timeout=120)
        response.raise_for_status()
        raw, source = response.content, GSM8K_URL
    dataset = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if len(dataset) != 1319:
        raise ValueError("expected all 1319 official GSM8K test questions")
    topics = [
        "astronomy",
        "gardening",
        "geometry",
        "history",
        "transport",
        "cooking",
        "music",
        "biology",
        "weather",
        "software",
        "literature",
        "architecture",
        "geology",
        "energy",
        "painting",
        "language",
        "ecology",
        "physics",
    ]
    prompts, source_texts = {"1": [], "8192": []}, []
    for index, topic in enumerate(topics):
        text = "\n\n".join(
            f"{topic.capitalize()} notebook {index + 1}, observation {step + 1}. "
            f"The team recorded {17 + step * 3} samples over {2 + step % 7} days. "
            "Compare the observations, explain possible causes, and identify which "
            "additional measurements would distinguish the explanations. Summarize "
            "the evidence in plain language with a worked example."
            for step in range(320)
        )
        tokens = tokenize(args.base_url, args.model, prompt=text)
        if len(tokens) < 8192:
            raise ValueError("source text tokenizes to fewer than 8192 tokens")
        prompts["8192"].append(tokens[:8192])
        prompts["1"].append(tokenize(args.base_url, args.model, prompt=topic)[:1])
        source_texts.append(text)
    cases = []
    template_kwargs = {"thinking_effort": "low"}
    for index, row in enumerate(dataset[:100]):
        messages = [
            {
                "role": "user",
                "content": "Solve this math problem. Show concise work and end with "
                "'#### ' followed by only the numeric answer.\n\n" + row["question"],
            }
        ]
        gold = answer(row["answer"])
        if gold is None:
            raise ValueError(f"GSM8K gold answer missing at index {index}")
        cases.append(
            {
                "index": index,
                "question": row["question"],
                "gold": gold,
                "messages": messages,
                "prompt_ids": tokenize(
                    args.base_url,
                    args.model,
                    messages=messages,
                    add_generation_prompt=True,
                    chat_template_kwargs=template_kwargs,
                ),
            }
        )
    save(
        args.output,
        {
            "schema_version": 1,
            "model": args.model,
            "performance_prompts": prompts,
            "performance_source_texts": source_texts,
            "short_source_texts": topics,
            "warmup_indices": list(range(10, 18)),
            "measured_indices": list(range(10)),
            "gsm8k": {
                "source": source,
                "dataset_sha256": hashlib.sha256(raw).hexdigest(),
                "selection": "first 100 of the official 1319-row test split",
                "chat_template_kwargs": template_kwargs,
                "cases": cases,
            },
        },
    )


def load_inputs(path, model):
    value = json.loads(path.read_text())
    if value.get("schema_version") != 1 or value.get("model") != model:
        raise ValueError("cached input schema/model mismatch")
    for size in (1, 8192):
        prompts = value["performance_prompts"][str(size)]
        if len(prompts) != 18 or any(len(prompt) != size for prompt in prompts):
            raise ValueError("cached performance prompt lengths differ")
        if any(type(token) is not int or token < 0 for p in prompts for token in p):
            raise ValueError("invalid cached performance token IDs")
    cases = value["gsm8k"]["cases"]
    if [case["index"] for case in cases] != list(range(100)):
        raise ValueError("expected the same fixed first 100 GSM8K questions")
    if any(
        not c["prompt_ids"]
        or c["gold"] is None
        or any(type(token) is not int or token < 0 for token in c["prompt_ids"])
        for c in cases
    ):
        raise ValueError("invalid cached GSM8K cases")
    return value


def quantiles(values):
    values = sorted(values)
    if not values:
        return None

    def percentile(q):
        position = (len(values) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    return {
        "mean": statistics.mean(values),
        "p10": percentile(0.1),
        "median": statistics.median(values),
        "p90": percentile(0.9),
    }


def summarize_requests(rows):
    good = [row for row in rows if "error" not in row]
    elapsed = max(row["ended"] for row in rows) - min(row["started"] for row in rows)
    return {
        "requested": len(rows),
        "completed": len(good),
        "failures": len(rows) - len(good),
        "wall_seconds": elapsed,
        "output_tokens_per_second": sum(len(row["token_ids"]) for row in good)
        / elapsed,
        "ttft_ms": quantiles([row["ttft_ms"] for row in good]),
        "tpot_ms": quantiles(
            [row["tpot_ms"] for row in good if row["tpot_ms"] is not None]
        ),
        "e2e_ms": quantiles([row["e2e_ms"] for row in good]),
        "multi_token_chunk_requests": sum(
            not row["single_token_chunks"] for row in good
        ),
    }


def clock_anchor():
    before = time.perf_counter()
    unix = time.time()
    after = time.perf_counter()
    return {
        "unix": unix,
        "monotonic": (before + after) / 2,
        "clock_uncertainty_seconds": (after - before) / 2,
    }


def worker_snapshot(root, phase, *, timeout=60):
    request_id = str(uuid.uuid4())
    root = Path(root)
    save(
        root / "snapshot-request.json",
        {"request_id": request_id, "phase": phase, "unix": time.time()},
    )
    path = root / f"snapshot-{request_id}.json"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            value = json.loads(path.read_text())
            if value.get("request_id") != request_id:
                raise ValueError("worker snapshot request ID mismatch")
            if not isinstance(value.get("unix"), (int, float)) or not isinstance(
                value.get("calls"), dict
            ):
                raise ValueError("worker snapshot lacks its timestamp/call counts")
            return {
                "request_id": request_id,
                "path": str(path),
                "sha256": digest(path),
                "unix": value["unix"],
            }
        time.sleep(0.1)
    raise TimeoutError(f"rank-zero snapshot was not acknowledged: {request_id}")


def metrics(base_url):
    requested = clock_anchor()
    try:
        response = requests.get(base_url.rstrip("/") + "/metrics", timeout=30)
        response.raise_for_status()
        return {**clock_anchor(), "requested": requested, "text": response.text}
    except Exception as error:
        return {**clock_anchor(), "requested": requested, "error": repr(error)}


def run(args):
    inputs = load_inputs(args.inputs, args.model)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "requests.jsonl"
    if path.exists() or (args.output / "summary.json").exists():
        raise ValueError("use a fresh output directory; preserve failed attempts")
    metadata = json.loads(args.server_metadata.read_text())
    if metadata.get("arm") != ARMS[args.slot]:
        raise ValueError("server receipt arm does not match the ABBA slot")
    for key in ("start_id", "source_commit", "compile_cache"):
        if not isinstance(metadata.get(key), str) or not metadata[key]:
            raise ValueError(f"server receipt requires a nonempty {key}")
    result = {
        "schema_version": 1,
        "slot": args.slot,
        "arm": ARMS[args.slot],
        "model": args.model,
        "inputs_sha256": digest(args.inputs),
        "server_metadata": metadata,
        "server_metadata_sha256": digest(args.server_metadata),
        "performance_settings": PERFORMANCE_SETTINGS,
        "sampling": SAMPLING,
        "quality_output_tokens": args.quality_output_tokens,
        "cells": [],
        "phases": [],
        "worker_evidence_dir": str(args.worker_evidence_dir)
        if args.worker_evidence_dir
        else None,
        "quality": None,
        "complete": False,
        "started_unix": time.time(),
    }
    save(args.output / "summary.json", result)
    phase_path = args.output / "phases.jsonl"
    with path.open("x") as output, phase_path.open("x") as phases:

        def checkpoint(phase, edge, isl, concurrency, wave):
            label = (
                f"slot-{args.slot}/{phase}/isl-{isl}/c-{concurrency}/wave-{wave}/{edge}"
            )
            marker = {
                "slot": args.slot,
                "arm": ARMS[args.slot],
                "phase": phase,
                "edge": edge,
                "isl": isl,
                "concurrency": concurrency,
                "wave": wave,
            }
            if edge == "end":
                marker.update(clock_anchor())
            try:
                if args.worker_evidence_dir:
                    marker["worker_snapshot"] = worker_snapshot(
                        args.worker_evidence_dir, label
                    )
                if edge == "start":
                    marker.update(clock_anchor())
            except Exception as error:
                marker.update(clock_anchor(), error=repr(error))
                result["error"] = f"worker evidence failed: {error}"
                result["ended_unix"] = time.time()
                result["requests_sha256"] = digest(path)
                raise
            finally:
                result["phases"].append(marker)
                phases.write(json.dumps(marker, allow_nan=False) + "\n")
                phases.flush()
                result["phases_sha256"] = digest(phase_path)
                save(args.output / "summary.json", result)

        def batch(cases, concurrency, phase, *, isl=None, wave=None):
            rows = []

            def one(case):
                row = request(
                    args.base_url,
                    args.model,
                    case["prompt_ids"],
                    args.quality_output_tokens if phase == "quality" else 1024,
                    exact=phase != "quality",
                    timeout=args.timeout,
                )
                row.update(
                    index=case["index"], phase=phase, concurrency=concurrency, isl=isl
                )
                if phase == "quality":
                    row["gold"] = case["gold"]
                    row["prediction"] = answer(row.get("text", ""))
                    row["correct"] = row["prediction"] == row["gold"]
                return row

            checkpoint(phase, "start", isl, concurrency, wave)
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                for future in as_completed([pool.submit(one, case) for case in cases]):
                    row = future.result()
                    output.write(json.dumps(row, allow_nan=False) + "\n")
                    output.flush()
                    rows.append(row)
            checkpoint(phase, "end", isl, concurrency, wave)
            return rows

        for isl, concurrency in CELLS:
            prompts = inputs["performance_prompts"][str(isl)]
            warmup = []
            for wave in range(2):
                cases = [
                    {"index": index, "prompt_ids": prompts[index]}
                    for index in range(
                        10 + wave * concurrency, 10 + (wave + 1) * concurrency
                    )
                ]
                warmup.extend(batch(cases, concurrency, "warmup", isl=isl, wave=wave))
            cell = {
                "isl": isl,
                "concurrency": concurrency,
                "warmup": summarize_requests(warmup),
            }
            result["cells"].append(cell)
            if cell["warmup"]["failures"]:
                result["error"] = "warmup failed; measurement was not started"
                break
            before = metrics(args.base_url)
            rows = batch(
                [{"index": i, "prompt_ids": prompts[i]} for i in range(10)],
                concurrency,
                "measured",
                isl=isl,
            )
            after = metrics(args.base_url)
            save(
                args.output / f"metrics-{isl}-{concurrency}.json",
                {"before": before, "after": after},
            )
            metrics_path = args.output / f"metrics-{isl}-{concurrency}.json"
            cell["metrics"] = {
                "path": str(metrics_path),
                "sha256": digest(metrics_path),
                "before_unix": before["unix"],
                "after_unix": after["unix"],
            }
            cell["measured"] = summarize_requests(rows)
            save(args.output / "summary.json", result)
            print(json.dumps(cell), flush=True)
        if args.accuracy and "error" not in result:
            rows = batch(inputs["gsm8k"]["cases"], 4, "quality")
            result["quality"] = {
                "requested": len(rows),
                "completed": sum("error" not in row for row in rows),
                "failures": sum("error" in row for row in rows),
                "correct": sum(row["correct"] for row in rows),
                "accuracy": sum(row["correct"] for row in rows) / len(rows),
                "unparsed_answers": sum(row["prediction"] is None for row in rows),
                "length_truncations": sum(
                    row.get("finish_reason") == "length" for row in rows
                ),
            }
    result["ended_unix"] = time.time()
    result["requests_sha256"] = digest(path)
    result["complete"] = (
        len(result["cells"]) == 4
        and all(
            "measured" in cell and cell["measured"]["failures"] == 0
            for cell in result["cells"]
        )
        and (
            not args.accuracy
            or result["quality"] is not None
            and result["quality"]["failures"] == 0
        )
    )
    save(args.output / "summary.json", result)
    if not result["complete"]:
        raise RuntimeError(
            "serving run failed; retained requests and summary identify failures"
        )


def summarize(args):
    runs = [json.loads((path / "summary.json").read_text()) for path in args.runs]
    runs.sort(key=lambda run: run["slot"])
    if [run["slot"] for run in runs] != [0, 1, 2, 3]:
        raise ValueError("supply exactly slots 0, 1, 2, 3 in ABBA order")
    if any(run["arm"] != ARMS[i] for i, run in enumerate(runs)):
        raise ValueError("ABBA arm mismatch")
    for key in (
        "model",
        "inputs_sha256",
        "performance_settings",
        "sampling",
        "quality_output_tokens",
    ):
        if any(run[key] != runs[0][key] for run in runs):
            raise ValueError(f"campaign mismatch: {key}")
    for key in ("start_id", "compile_cache"):
        if len({run["server_metadata"][key] for run in runs}) != 4:
            raise ValueError(f"fresh ABBA starts require four distinct {key} values")
    if any(runs[i]["started_unix"] < runs[i - 1]["ended_unix"] for i in range(1, 4)):
        raise ValueError("ABBA client runs overlapped or were executed out of order")
    by_slot = {}
    for path in args.runs:
        summary = json.loads((path / "summary.json").read_text())
        if (
            summary.get("phases_sha256")
            and digest(path / "phases.jsonl") != summary["phases_sha256"]
        ):
            raise ValueError("phase marker artifact hash mismatch")
        for marker in summary.get("phases", []):
            receipt = marker.get("worker_snapshot")
            if receipt and digest(receipt["path"]) != receipt["sha256"]:
                raise ValueError("worker snapshot artifact hash mismatch")
        if digest(path / "requests.jsonl") != summary["requests_sha256"]:
            raise ValueError("request artifact hash mismatch")
        by_slot[summary["slot"]] = [
            json.loads(line)
            for line in (path / "requests.jsonl").read_text().splitlines()
        ]
    report = {
        "schema_version": 1,
        "complete": all(run["complete"] for run in runs),
        "runs": runs,
        "cells": [],
        "quality_pairs": [],
    }
    for isl, concurrency in CELLS:
        values = []
        for run in runs:
            rows = [
                row
                for row in by_slot[run["slot"]]
                if row["phase"] == "measured"
                and row["isl"] == isl
                and row["concurrency"] == concurrency
            ]
            if sorted(row["index"] for row in rows) != list(range(10)):
                report["complete"] = False
            values.append(summarize_requests(rows) if rows else None)
        if any(value is None or value["failures"] for value in values):
            report["complete"] = False
            report["cells"].append(
                {"isl": isl, "concurrency": concurrency, "valid": False, "runs": values}
            )
            continue
        metrics_out = {}
        for metric, statistic in (
            ("ttft_ms", "median"),
            ("tpot_ms", "mean"),
            ("output_tokens_per_second", None),
        ):
            points = [
                value[metric][statistic] if statistic else value[metric]
                for value in values
            ]
            native, custom = (
                statistics.mean((points[0], points[3])),
                statistics.mean((points[1], points[2])),
            )
            metrics_out[metric] = {
                "baseline": native,
                "candidate": custom,
                "delta_percent": 100 * (custom / native - 1),
                "pair_delta_percent": [
                    100 * (points[1] / points[0] - 1),
                    100 * (points[2] / points[3] - 1),
                ],
            }
        report["cells"].append(
            {
                "isl": isl,
                "concurrency": concurrency,
                "valid": True,
                "runs": values,
                "comparison": metrics_out,
            }
        )
    for a, b in ((0, 1), (3, 2)):
        left = {row["index"]: row for row in by_slot[a] if row["phase"] == "quality"}
        right = {row["index"]: row for row in by_slot[b] if row["phase"] == "quality"}
        if not left and not right:
            continue
        if set(left) != set(range(100)) or set(right) != set(range(100)):
            raise ValueError("accuracy comparisons require matching 100-question arms")
        paired = []
        for index in range(100):
            x, y = left[index], right[index]
            paired.append(
                {
                    "index": index,
                    "baseline_correct": x["correct"],
                    "candidate_correct": y["correct"],
                    "answer_equal": x["prediction"] is not None
                    and x["prediction"] == y["prediction"],
                    "tokens_equal": bool(x.get("token_ids"))
                    and x.get("token_ids") == y.get("token_ids"),
                    "request_failed": "error" in x or "error" in y,
                }
            )
        report["quality_pairs"].append(
            {
                "slots": [a, b],
                "baseline_correct": sum(row["baseline_correct"] for row in paired),
                "candidate_correct": sum(row["candidate_correct"] for row in paired),
                "answer_disagreements": sum(not row["answer_equal"] for row in paired),
                "token_disagreements": sum(not row["tokens_equal"] for row in paired),
                "paired": paired,
            }
        )
    report["performance_complete"] = report["complete"]
    report["accuracy_complete"] = bool(report["quality_pairs"]) and all(
        not row["request_failed"]
        for pair in report["quality_pairs"]
        for row in pair["paired"]
    )
    report["complete"] = report["performance_complete"] and report["accuracy_complete"]
    save(args.output, report)
    markdown = [
        "# Two Shot serving comparison",
        "",
        "Each arm is the mean of two fresh-start statistics; requests are not pooled.",
        "",
        "| ISL | Concurrency | Metric | Baseline | Candidate | Delta |",
        "| ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for cell in report["cells"]:
        if cell["valid"]:
            for metric, value in cell["comparison"].items():
                markdown.append(
                    f"| {cell['isl']} | {cell['concurrency']} | {metric} | "
                    f"{value['baseline']:.3f} | {value['candidate']:.3f} | "
                    f"{value['delta_percent']:+.2f}% |"
                )
    markdown.extend(
        [
            "",
            f"Complete performance campaign: {report['performance_complete']}.",
            f"Complete paired accuracy check: {report['accuracy_complete']}.",
            "",
            "Per-start spreads, failures and raw samples remain in the JSON report.",
            "Review server execution evidence and matched launch settings separately.",
        ]
    )
    markdown.extend(
        [
            "",
            "| GSM8K slots | Baseline / 100 | Candidate / 100 | Answer differences |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for pair in report["quality_pairs"]:
        markdown.append(
            f"| {pair['slots']} | {pair['baseline_correct']} | "
            f"{pair['candidate_correct']} | {pair['answer_disagreements']} |"
        )
    args.output.with_suffix(".md").write_text("\n".join(markdown) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--base-url", required=True)
    prep.add_argument("--model", required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument(
        "--dataset", type=Path, help="official GSM8K test.jsonl (1319 rows)"
    )
    measure = sub.add_parser("run")
    measure.add_argument("--base-url", required=True)
    measure.add_argument("--model", required=True)
    measure.add_argument("--inputs", type=Path, required=True)
    measure.add_argument("--output", type=Path, required=True)
    measure.add_argument("--server-metadata", type=Path, required=True)
    measure.add_argument("--slot", type=int, choices=range(4), required=True)
    measure.add_argument("--accuracy", action="store_true")
    measure.add_argument("--worker-evidence-dir", type=Path)
    measure.add_argument("--quality-output-tokens", type=int, default=1024)
    measure.add_argument("--timeout", type=float, default=1800)
    summary = sub.add_parser("summarize")
    summary.add_argument("--runs", type=Path, nargs=4, required=True)
    summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run" and (args.quality_output_tokens < 1 or args.timeout <= 0):
        parser.error("quality output tokens and timeout must be positive")
    {"prepare": prepare, "run": run, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()
