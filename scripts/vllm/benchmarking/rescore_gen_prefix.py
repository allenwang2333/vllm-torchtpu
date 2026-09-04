# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Re-score an assistant-prefill lm-eval task on the answer it contains.

``mmlu_llama`` sets ``gen_prefix: "The best answer is"``, so lm-eval hands the
model a partially written assistant turn and exact-matches only the
continuation (``"B."``). A server that cannot leave that turn open -- Kimi-K3
cannot, see ``vllm_torchtpu/tokenizers.py`` -- makes the model restate the stem,
and ``strict_match`` reads ~0 for a model that is answering correctly. The
responses still carry the letter, so the run is recoverable without re-serving:
that is what this script does, and it works on samples already on disk.

It writes a **separate** task key (``<task>_tpu``) with its own baseline. It is
not a redefinition of ``mmlu_llama``: letter-extraction and strict-continuation
measure different things, and the existing ``mmlu_llama`` baselines were
calibrated on the latter.

The extraction is the task's own contract -- its prompt says *Your response
should end with "The best answer is [the_answer_letter]"* -- and is the same
regex ``mmlu_pro`` already applies. Responses that need the positional fallback
are counted and reported rather than folded in silently: a score whose
fallback rate is not ~0 is not a score you should quote.

Usage:
    python3 rescore_gen_prefix.py --results-dir DIR [--task mmlu_llama]
"""
import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

# The phrasing the task instructs the model to end with, as mmlu_pro matches it.
PRIMARY_RE = re.compile(r"[Tt]he best answer is\s*\(?([A-Z])\)?")
# Last standalone capital letter, for a response that answered without the
# carrier phrase. Counted separately; see the module docstring.
FALLBACK_RE = re.compile(r"\b([A-Z])\b")
TARGET_RE = re.compile(r"([A-Z])")

FILTER_NAME = "letter-extract"


def extract_letter(response: str) -> tuple[str | None, str]:
    """Return (letter, how) where how is 'primary', 'fallback' or 'none'."""
    if hits := PRIMARY_RE.findall(response):
        return hits[-1], "primary"
    if hits := FALLBACK_RE.findall(response):
        return hits[-1], "fallback"
    return None, "none"


def response_of(sample: dict) -> str:
    """The model's text, preferring the raw response over a filtered one.

    ``filtered_resps`` has already been through the task's own filter, which is
    exactly the stage this script is replacing.
    """
    for key in ("resps", "filtered_resps"):
        value = sample.get(key)
        while isinstance(value, list) and value:
            value = value[0]
        if isinstance(value, str):
            return value
    raise KeyError(f"no response text in sample {sample.get('doc_id')!r}")


def score_file(path: Path) -> tuple[int, int, Counter]:
    correct = total = 0
    how = Counter()
    # Iterate the handle, not str.splitlines(): JSON permits raw U+2028/U+2029
    # inside strings and splitlines() would break such a record in two.
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        sample = json.loads(line)
        target = TARGET_RE.search(str(sample["target"]))
        if target is None:
            raise ValueError(
                f"{path.name}: target {sample['target']!r} has no letter")
        letter, kind = extract_letter(response_of(sample))
        how[kind] += 1
        total += 1
        correct += letter == target.group(1)
    return correct, total, how


def subject_of(path: Path, task: str) -> str:
    """``samples_<group>::<subject>_<timestamp>.jsonl`` -> ``<subject>``."""
    stem = path.stem.removeprefix("samples_")
    subject = stem.split("::")[-1] if "::" in stem else stem
    return re.sub(r"_\d{4}-\d{2}-\d{2}T[\d.-]+$", "", subject) or task


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--task", default="mmlu_llama")
    ap.add_argument("--suffix", default="_tpu")
    ap.add_argument("--max-fallback-rate",
                    type=float,
                    default=0.02,
                    help="fail if more than this fraction of responses need "
                    "the positional fallback")
    args = ap.parse_args(argv)

    files = sorted(args.results_dir.rglob(f"samples_{args.task}*.jsonl"))
    if not files:
        print(
            f"::error::no samples_{args.task}*.jsonl under {args.results_dir}; "
            f"lm_eval needs --log_samples",
            file=sys.stderr)
        return 1

    per_subject: dict[str, float] = {}
    correct = total = 0
    how: Counter = Counter()
    for path in files:
        c, n, h = score_file(path)
        if n:
            per_subject[subject_of(path, args.task)] = c / n
        correct, total, how = correct + c, total + n, how + h

    if not total:
        print(
            f"::error::{len(files)} sample files under {args.results_dir} "
            f"held no rows",
            file=sys.stderr)
        return 1

    acc = correct / total
    stderr = math.sqrt(acc * (1 - acc) / total)
    task_name = f"{args.task}{args.suffix}"
    out = {
        "results": {
            task_name: {
                f"exact_match,{FILTER_NAME}": acc,
                f"exact_match_stderr,{FILTER_NAME}": stderr,
                "sample_len": float(total),
            }
        },
        "rescore": {
            "source_task": args.task,
            "files": len(files),
            "correct": correct,
            "total": total,
            "extraction": dict(how),
            "per_subject": per_subject,
        },
    }
    dest = args.results_dir / f"results_{task_name}.json"
    dest.write_text(json.dumps(out, indent=2, sort_keys=True))

    fallback_rate = how["fallback"] / total
    print(f"{task_name}: exact_match,{FILTER_NAME}={acc:.4f} "
          f"({correct}/{total}, stderr {stderr:.5f}) -> {dest}")
    print(f"  extraction: {dict(how)}  fallback_rate={fallback_rate:.4%}")
    if how["none"]:
        print(
            f"::error::{how['none']} responses carried no letter at all; "
            f"the score is not trustworthy",
            file=sys.stderr)
        return 1
    if fallback_rate > args.max_fallback_rate:
        print(
            f"::error::{fallback_rate:.2%} of responses lacked the carrier "
            f"phrase (limit {args.max_fallback_rate:.2%}); the model is not "
            f"following the prompt and the score is not comparable",
            file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
