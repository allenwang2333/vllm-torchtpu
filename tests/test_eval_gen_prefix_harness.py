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
"""The eval harness half of the assistant-prefill fix.

Two invariants, both of which `run_eval_flow.sh` violated before this change:
``continue_final_message`` goes only to tasks that prefill an assistant turn,
and a task that does prefill one can still be scored when the server renders
it closed.
"""
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "scripts" / "vllm" / "benchmarking"
FLOW = BENCH / "run_eval_flow.sh"


@pytest.fixture(scope="module")
def rescore():
    """Loaded lazily so the two `run_eval_flow.sh` tests below still run on a
    tree where this script does not exist yet."""
    spec = importlib.util.spec_from_file_location(
        "rescore_gen_prefix", BENCH / "rescore_gen_prefix.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------- run_eval_flow


def _prefill_case_block() -> str:
    """The `case` that decides whether a task prefills an assistant turn."""
    lines = FLOW.read_text().splitlines()
    for start, line in enumerate(lines):
        if line.strip().startswith("case \"$task\" in") and any(
                "prefills_assistant_turn" in nxt
                for nxt in lines[start:start + 8]):
            end = next(i for i in range(start, len(lines))
                       if lines[i].strip() == "esac")
            return "\n".join(lines[start:end + 1])
    raise AssertionError(
        "run_eval_flow.sh has no `case` selecting prefills_assistant_turn; "
        "continue_final_message is therefore not gated on the task")


@pytest.mark.parametrize("task,prefills", [
    ("mmlu_llama", "1"),
    ("mmlu_llama_de", "1"),
    ("arc_challenge_llama", "1"),
    ("mmlu_pro", "0"),
    ("humaneval_plus_tpu", "0"),
    ("gsm8k", "0"),
])
def test_only_prefill_tasks_are_flagged(task, prefills):
    script = f'task="{task}"\n{_prefill_case_block()}\nprintf %s "$prefills_assistant_turn"'
    done = subprocess.run(["bash", "-c", script],
                          capture_output=True,
                          text=True,
                          check=True)
    assert done.stdout == prefills


def test_continue_final_message_is_branch_guarded():
    """The flag must sit under the prefill test, not in a catch-all `else`."""
    lines = FLOW.read_text().splitlines()
    sites = [
        i for i, ln in enumerate(lines)
        if "continue_final_message" in ln and "--gen_kwargs" in ln
    ]
    assert sites, "expected run_eval_flow.sh to still pass the flag somewhere"
    for site in sites:
        guard = next(lines[i] for i in range(site, -1, -1)
                     if re.match(r"\s*(if|elif|else)\b", lines[i]))
        assert "prefills_assistant_turn" in guard, (
            f"line {site + 1} sends continue_final_message under {guard.strip()!r}; "
            "on a user-final prompt that strips the user turn's terminator")


# ---------------------------------------------------------- rescore_gen_prefix


def _sample(target, response, exact_match=0):
    return json.dumps({
        "doc_id": 0,
        "target": target,
        "resps": [[response]],
        "filtered_resps": [[response.strip()]],
        "exact_match": exact_match,
    })


def _write(tmp_path, rows, task="mmlu_llama"):
    path = tmp_path / f"samples_{task}_humanities::{task}_logic_2026-01-01T00-00-00.0.jsonl"
    path.write_text("\n".join(rows) + "\n")
    return path


def test_restated_stem_scores_as_correct(rescore, tmp_path):
    """The exact shape lm-eval reads as 0: a restated stem carrying the letter."""
    rows = [
        _sample("D.", " The best answer is D"),
        _sample("B.", " The best answer is B")
    ]
    _write(tmp_path, rows)
    assert rescore.main(["--results-dir", str(tmp_path)]) == 0

    written = json.loads(
        (tmp_path / "results_mmlu_llama_tpu.json").read_text())
    assert written["results"]["mmlu_llama_tpu"][
        "exact_match,letter-extract"] == 1.0
    # ...while every one of those samples was scored 0 by the task's own filter.
    assert all(json.loads(r)["exact_match"] == 0 for r in rows)
    assert written["rescore"]["extraction"] == {"primary": 2}


def test_wrong_answers_still_score_wrong(rescore, tmp_path):
    """A scorer that cannot fail is not a scorer."""
    _write(tmp_path, [
        _sample("A.", "The best answer is D"),
        _sample("B.", "The best answer is B")
    ])
    assert rescore.main(["--results-dir", str(tmp_path)]) == 0
    written = json.loads(
        (tmp_path / "results_mmlu_llama_tpu.json").read_text())
    assert written["results"]["mmlu_llama_tpu"][
        "exact_match,letter-extract"] == 0.5


def test_a_letterless_response_fails_the_run(rescore, tmp_path, capsys):
    _write(tmp_path, [_sample("A.", "i cannot answer that")])
    assert rescore.main(["--results-dir", str(tmp_path)]) == 1
    assert "no letter at all" in capsys.readouterr().err


def test_too_many_fallbacks_fails_the_run(rescore, tmp_path, capsys):
    """Answers without the carrier phrase are counted, not quietly accepted."""
    _write(
        tmp_path,
        [_sample("A.", "The best answer is A"),
         _sample("B.", "probably B")])
    assert rescore.main(["--results-dir", str(tmp_path)]) == 1
    assert "lacked the carrier phrase" in capsys.readouterr().err


def test_missing_samples_is_an_error_not_an_empty_score(
        rescore, tmp_path, capsys):
    assert rescore.main(["--results-dir", str(tmp_path)]) == 1
    assert "--log_samples" in capsys.readouterr().err


@pytest.mark.parametrize("response,letter,how", [
    (" The best answer is D", "D", "primary"),
    ("the best answer is (C).", "C", "primary"),
    ("The best answer is A, no wait. The best answer is B.", "B", "primary"),
    ("I think it is C", "C", "fallback"),
    ("no idea", None, "none"),
])
def test_letter_extraction(rescore, response, letter, how):
    assert rescore.extract_letter(response) == (letter, how)


def test_metric_key_is_gated_by_check_regression(rescore):
    """A score no regression check knows how to read is not a gate."""
    spec = importlib.util.spec_from_file_location(
        "check_regression", BENCH / "check_regression.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_regression"] = module
    spec.loader.exec_module(module)
    assert f"exact_match,{rescore.FILTER_NAME}" in module.ACC_KEYS
    assert f"exact_match_stderr,{rescore.FILTER_NAME}" in module.STDERR_KEYS
