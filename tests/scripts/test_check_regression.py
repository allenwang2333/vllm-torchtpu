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
"""Unit tests for scripts/vllm/benchmarking/check_regression.py."""

import importlib.util
import json
import pathlib
import sys

_SCRIPT = (pathlib.Path(__file__).resolve().parents[2] / "scripts" / "vllm" /
           "benchmarking" / "check_regression.py")
_spec = importlib.util.spec_from_file_location("check_regression", _SCRIPT)
check_regression = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_regression)


def test_primary_metric():
    entry = {
        "exact_match,strict_match": 0.885,
        "exact_match_stderr,strict_match": 0.010,
    }
    metric = check_regression.primary_metric(entry)
    assert metric == ("exact_match,strict_match", 0.885)


def test_primary_stderr_matching():
    entry = {
        "exact_match,strict_match": 0.885,
        "exact_match_stderr,strict_match": 0.012,
    }
    stderr = check_regression.primary_stderr(entry)
    assert stderr == 0.012


def test_primary_stderr_custom_extract():
    entry = {
        "exact_match,custom-extract": 0.807,
        "exact_match_stderr,custom-extract": 0.024,
    }
    stderr = check_regression.primary_stderr(entry)
    assert stderr == 0.024


def test_primary_stderr_fallback():
    entry = {
        "acc_stderr,none": 0.018,
    }
    stderr = check_regression.primary_stderr(entry)
    assert stderr == 0.018


def test_primary_stderr_absent():
    entry = {
        "exact_match,strict_match": 0.885,
    }
    stderr = check_regression.primary_stderr(entry)
    assert stderr is None


def test_check_eval_pass_exact():
    baseline = {
        "mmlu_llama": {
            "exact_match,strict_match": 0.88,
            "exact_match_stderr,strict_match": 0.01,
        }
    }
    results = {
        "mmlu_llama": {
            "exact_match,strict_match": 0.88,
        }
    }
    failed = check_regression.check_eval(results, baseline, tolerance=0.02)
    assert failed == 0


def test_check_eval_scales_with_stderr():
    # baseline has stderr = 0.035. With tolerance = 0.02, allowed drop is 0.035.
    baseline = {
        "mmlu_llama_other": {
            "exact_match,strict_match": 0.88,
            "exact_match_stderr,strict_match": 0.035,
        }
    }
    # drop of 0.030 (actual 0.85) -> passes under max(0.02, 0.035) = 0.035
    results_pass = {
        "mmlu_llama_other": {
            "exact_match,strict_match": 0.85,
        }
    }
    assert check_regression.check_eval(results_pass, baseline,
                                       tolerance=0.02) == 0

    # drop of 0.040 (actual 0.84) -> fails (> 0.035)
    results_fail = {
        "mmlu_llama_other": {
            "exact_match,strict_match": 0.84,
        }
    }
    assert check_regression.check_eval(results_fail, baseline,
                                       tolerance=0.02) == 1


def test_check_eval_uses_tolerance_floor_when_stderr_is_smaller():
    # baseline has stderr = 0.008. With tolerance = 0.02, allowed drop is 0.02 (floor).
    baseline = {
        "mmlu_llama": {
            "exact_match,strict_match": 0.88,
            "exact_match_stderr,strict_match": 0.008,
        }
    }
    # drop of 0.015 -> passes (within 0.02 floor)
    results_pass = {
        "mmlu_llama": {
            "exact_match,strict_match": 0.865,
        }
    }
    assert check_regression.check_eval(results_pass, baseline,
                                       tolerance=0.02) == 0

    # drop of 0.025 -> fails (> 0.02 floor)
    results_fail = {
        "mmlu_llama": {
            "exact_match,strict_match": 0.855,
        }
    }
    assert check_regression.check_eval(results_fail, baseline,
                                       tolerance=0.02) == 1


def test_check_eval_without_stderr():
    # When no stderr is provided, tolerance floor is used strictly
    baseline = {
        "mmlu_llama": {
            "exact_match,strict_match": 0.88,
        }
    }
    results = {
        "mmlu_llama": {
            "exact_match,strict_match": 0.865,
        }
    }
    # delta is -0.015, tolerance is 0.02 -> PASS
    assert check_regression.check_eval(results, baseline, tolerance=0.02) == 0

    # delta is -0.025, tolerance is 0.02 -> FAIL
    results["mmlu_llama"]["exact_match,strict_match"] = 0.855
    assert check_regression.check_eval(results, baseline, tolerance=0.02) == 1


def test_check_regression_cli(tmp_path, monkeypatch):
    baseline_file = tmp_path / "test.baseline.json"
    baseline_data = {
        "mmlu_llama": {
            "exact_match,strict_match": 0.88,
            "exact_match_stderr,strict_match": 0.01,
        },
        "mmlu_llama_other": {
            "exact_match,strict_match": 0.88,
            "exact_match_stderr,strict_match": 0.035,
        }
    }
    with baseline_file.open("w") as f:
        json.dump(baseline_data, f)

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    results_file = results_dir / "results_mmlu_llama.json"
    # drop on mmlu_llama is 0.01 (within 0.02 floor)
    # drop on mmlu_llama_other is 0.03 (within 0.035 stderr)
    results_data = {
        "results": {
            "mmlu_llama": {
                "exact_match,strict_match": 0.87,
                "exact_match_stderr,strict_match": 0.01,
            },
            "mmlu_llama_other": {
                "exact_match,strict_match": 0.85,
                "exact_match_stderr,strict_match": 0.035,
            }
        }
    }
    with results_file.open("w") as f:
        json.dump(results_data, f)

    # CLI test passing with default tolerance (0.015)
    monkeypatch.setattr(sys, "argv", [
        "check_regression.py",
        "--mode",
        "eval",
        "--results-dir",
        str(results_dir),
        "--baseline",
        str(baseline_file),
    ])
    ret = check_regression.main()
    assert ret == 0

    # CLI test failing when other drops beyond its stderr (0.035)
    results_data["results"]["mmlu_llama_other"][
        "exact_match,strict_match"] = 0.84
    with results_file.open("w") as f:
        json.dump(results_data, f)

    monkeypatch.setattr(sys, "argv", [
        "check_regression.py",
        "--mode",
        "eval",
        "--results-dir",
        str(results_dir),
        "--baseline",
        str(baseline_file),
    ])
    ret = check_regression.main()
    assert ret == 1


if __name__ == "__main__":
    import tempfile

    test_primary_metric()
    test_primary_stderr_matching()
    test_primary_stderr_custom_extract()
    test_primary_stderr_fallback()
    test_primary_stderr_absent()
    test_check_eval_pass_exact()
    test_check_eval_scales_with_stderr()
    test_check_eval_uses_tolerance_floor_when_stderr_is_smaller()
    test_check_eval_without_stderr()

    class MonkeyPatch:

        def setattr(self, target, attr, value):
            setattr(target, attr, value)

    with tempfile.TemporaryDirectory() as td:
        test_check_regression_cli(pathlib.Path(td), MonkeyPatch())

    print("All unit tests passed successfully!")
