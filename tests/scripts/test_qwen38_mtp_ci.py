# SPDX-License-Identifier: Apache-2.0

import copy
import importlib.util
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts/vllm/integration/check_qwen38_mtp.py"
)
_spec = importlib.util.spec_from_file_location("qwen38_mtp_ci", _SCRIPT)
ci = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ci)


def counters(prefill=128, decode=128, drafts=100, proposed=300, accepted=240):
    return {
        "prefill": {"request_success": prefill},
        "decode": dict(
            zip(ci.COUNTERS, (decode, drafts, proposed, accepted), strict=True)
        ),
    }


def test_metrics_sum_series_and_accept_total_suffix():
    body = """# TYPE vllm:request_success counter
vllm:request_success_total{model_name="qwen38",engine="0"} 10
vllm:request_success_total{model_name="qwen38",engine="1"} 20
vllm:spec_decode_num_drafts{model_name="qwen38"} 40
vllm:request_success_created 1000
"""
    assert ci.metric_values(body) == {
        "request_success": 30,
        "spec_decode_num_drafts": 40,
    }


@pytest.mark.parametrize("value", ["NaN", "+Inf", "-Inf"])
def test_metrics_reject_nonfinite_values(value):
    with pytest.raises(ValueError, match="Non-finite"):
        ci.metric_values(f"vllm:request_success {value}")


@pytest.mark.parametrize("mode,prefill", [("pd", 128), ("direct", 0), ("chunked", 128)])
def test_path_uses_deltas_not_cumulative_activity(mode, prefill):
    before = counters()
    after = copy.deepcopy(before)
    after["prefill"]["request_success"] += prefill
    for name, delta in counters()["decode"].items():
        after["decode"][name] += delta
    result = ci.check_path(before, after, mode=mode, requests=128)
    assert result["acceptance_length"] == 3.4
    assert result["acceptance_rate"] == 0.8


@pytest.mark.parametrize(
    "mode,after,error",
    [
        ("pd", counters(prefill=0), "routing"),
        ("direct", counters(), "routing"),
        ("pd", counters(decode=127), "Incomplete"),
        ("pd", counters(drafts=0, proposed=0, accepted=0), "Invalid MTP"),
        ("pd", counters(accepted=301), "Invalid MTP"),
        ("pd", counters(accepted=100), "below 0.5"),
    ],
)
def test_path_rejects_bypass_incomplete_requests_and_inactive_mtp(mode, after, error):
    with pytest.raises(ValueError, match=error):
        ci.check_path(counters(0, 0, 0, 0, 0), after, mode=mode, requests=128)


@pytest.mark.parametrize(
    "pd,direct,passed",
    [
        (118 / 128, 120 / 128, True),
        (117 / 128, 120 / 128, False),
        (121 / 128, 120 / 128, True),
        (0.65, 0.65, False),
        (float("nan"), 0.9, False),
    ],
)
def test_accuracy_allows_bounded_run_to_run_variation(pd, direct, passed):
    assert ci.check_accuracy({"pd": pd, "direct": direct}, 128)["passed"] is passed


def test_evaluator_failure_propagates_and_preserves_log(tmp_path):
    import subprocess
    import sys

    log = tmp_path / "failed.log"
    with pytest.raises(subprocess.CalledProcessError):
        ci.run([sys.executable, "-c", "print('eval failed'); raise SystemExit(1)"], log)
    assert "eval failed" in log.read_text()
