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
"""Tests for structured decoding benchmark configs and regression metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.datasets.structured.schemas import COMPLEX_AST_SCHEMA, FLAT_JSON_SCHEMA
from scripts.vllm.benchmarking.check_regression import GATED_METRICS, check_perf


@pytest.mark.cpu_test
def test_structured_dataset_validity() -> None:
    dataset_file = (
        Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "datasets"
        / "structured"
        / "ast_schema_prompts.jsonl"
    )
    assert dataset_file.exists(), f"Missing dataset: {dataset_file}"

    lines = [
        line.strip() for line in dataset_file.read_text().splitlines() if line.strip()
    ]
    assert len(lines) >= 2, "Expected at least 2 prompt samples"

    for line in lines:
        entry = json.loads(line)
        assert "prompt" in entry
        assert "response_format" in entry
        rf = entry["response_format"]
        assert rf.get("type") == "json_schema"
        assert "json_schema" in rf


@pytest.mark.cpu_test
def test_schemas_module() -> None:
    assert FLAT_JSON_SCHEMA["type"] == "object"
    assert "properties" in FLAT_JSON_SCHEMA
    assert COMPLEX_AST_SCHEMA["type"] == "object"
    assert "$defs" in COMPLEX_AST_SCHEMA


@pytest.mark.cpu_test
def test_block_efficiency_in_gated_metrics() -> None:
    be_metrics = [m for m in GATED_METRICS if m.result_field == "block_efficiency"]
    assert len(be_metrics) == 1
    metric = be_metrics[0]
    assert metric.direction == "higher_is_better"


@pytest.mark.cpu_test
def test_regression_checker_evaluates_block_efficiency() -> None:
    baseline = {
        "isl512_osl512_c8": {
            "completed": 40,
            "median_tpot_ms": 3.10,
            "median_ttft_ms": 28.5,
            "output_token_throughput": 2580.0,
            "total_token_throughput": 5160.0,
            "block_efficiency": 3.84,
        }
    }

    # Case 1: Healthy run passing block_efficiency check (within 10% tolerance)
    healthy_results = {
        "isl512_osl512_c8": {
            "completed": 40,
            "median_tpot_ms": 3.10,
            "median_ttft_ms": 28.5,
            "output_token_throughput": 2580.0,
            "total_token_throughput": 5160.0,
            "block_efficiency": 3.75,
        }
    }
    assert check_perf(healthy_results, baseline, tolerance=0.10) == 0

    # Case 2: Regressed run failing block_efficiency check (dropped from 3.84 to 1.05)
    regressed_results = {
        "isl512_osl512_c8": {
            "completed": 40,
            "median_tpot_ms": 3.10,
            "median_ttft_ms": 28.5,
            "output_token_throughput": 2580.0,
            "total_token_throughput": 5160.0,
            "block_efficiency": 1.05,
        }
    }
    assert check_perf(regressed_results, baseline, tolerance=0.10) == 1
