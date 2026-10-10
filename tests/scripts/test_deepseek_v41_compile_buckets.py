# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for DeepSeek-V4.1 high-concurrency COMPILE_SIZES configs."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

CONFIGS_DIR = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "vllm"
    / "benchmarking"
    / "configs"
)


def _eval_config(script_name: str, env_overrides: dict[str, str]) -> dict[str, str]:
    script_path = CONFIGS_DIR / script_name
    cmd = (
        f"source {script_path} && "
        "python3 -c 'import json, os; print(json.dumps({"
        '"COMPILE_SIZES": os.environ.get("COMPILE_SIZES", ""), '
        '"MAX_NUM_SEQS": os.environ.get("MAX_NUM_SEQS", ""), '
        '"DATA_PARALLELISM": os.environ.get("DATA_PARALLELISM", ""), '
        '"TENSOR_PARALLELISM": os.environ.get("TENSOR_PARALLELISM", ""), '
        '"MAX_NUM_BATCHED_TOKENS": os.environ.get("MAX_NUM_BATCHED_TOKENS", ""), '
        '"SERVER_READY_WAIT_MIN": os.environ.get("SERVER_READY_WAIT_MIN", ""), '
        '"EXTRA_SERVE_ARGS": os.environ.get("EXTRA_SERVE_ARGS", "")'
        "}))'"
    )
    # Export sourced variables into child python3 environment via set -a.
    full_cmd = f"set -a && {cmd}"
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(
            (
                "MAX_NUM_SEQS",
                "MAX_NUM_BATCHED_TOKENS",
                "SERVER_READY_WAIT_MIN",
                "COMPILE_SIZES",
                "OFFLOAD",
                "HOST_DMA",
                "TPU_DEEPSEEK_V41_",
                "EXTRA_SERVE_ARGS",
                "KV_CACHE_MEMORY_BYTES",
                "CONCURRENCY_OPTIONS",
            )
        )
    }
    env.update(env_overrides)
    out = subprocess.check_output(["bash", "-c", full_cmd], env=env, text=True)
    return json.loads(out.strip().splitlines()[-1])


@pytest.mark.parametrize(
    ("script_name", "env_overrides", "expected_sizes"),
    [
        # TP8 DSpark K=5
        (
            "deepseek-v41-flash-tp8-ep-100k-decode-dspark.sh",
            {},
            "[4,8,16,24,48,96,2048]",
        ),
        (
            "deepseek-v41-flash-tp8-ep-100k-decode-dspark.sh",
            {"OFFLOAD": "1"},
            "[4,8,16,24,32,48,64,96,144,192,288,384,576,672,2048]",
        ),
        (
            "deepseek-v41-flash-tp8-ep-100k-decode-dspark.sh",
            {"HOST_DMA": "1", "MAX_NUM_SEQS": "224"},
            "[4,8,16,24,32,48,64,96,144,192,288,384,576,672,768,1152,1344,1536,2048]",
        ),
        # DP2 x TP4 DSpark K=5
        (
            "deepseek-v41-flash-dp2-tp4-ep-100k-decode-dspark.sh",
            {},
            "[4,8,16,24,48,96,2048]",
        ),
        (
            "deepseek-v41-flash-dp2-tp4-ep-100k-decode-dspark.sh",
            {"OFFLOAD": "1"},
            "[4,8,16,24,32,48,64,96,144,192,288,384,576,2048]",
        ),
        (
            "deepseek-v41-flash-dp2-tp4-ep-100k-decode-dspark.sh",
            {"OFFLOAD": "1", "MAX_NUM_SEQS": "224"},
            "[4,8,16,24,32,48,64,96,112,144,192,288,384,576,672,768,1152,1344,1536,2048]",
        ),
        # TP8 non-speculative decode
        (
            "deepseek-v41-flash-tp8-ep-100k-decode.sh",
            {},
            "[4,8,16,2048]",
        ),
        (
            "deepseek-v41-flash-tp8-ep-100k-decode.sh",
            {"MAX_NUM_SEQS": "224"},
            "[4,8,16,32,48,64,96,128,192,224,2048]",
        ),
        # DP2 x TP4 non-speculative decode
        (
            "deepseek-v41-flash-dp2-tp4-ep-100k-decode.sh",
            {},
            "[4,8,16,2048]",
        ),
        (
            "deepseek-v41-flash-dp2-tp4-ep-100k-decode.sh",
            {"OFFLOAD": "1"},
            "[4,8,16,24,32,48,64,96,128,192,224,2048]",
        ),
        (
            "deepseek-v41-flash-dp2-tp4-ep-100k-decode.sh",
            {"MAX_NUM_SEQS": "112"},
            "[4,8,16,24,32,48,64,96,128,192,224,2048]",
        ),
        (
            "deepseek-v41-flash-dp2-tp4-ep-100k-decode.sh",
            {"MAX_NUM_SEQS": "64"},
            "[4,8,16,32,48,64,96,128,192,224,2048]",
        ),
        # TP8 owner-sharded wrapper (drops 4 since TP=8 requires multiples of 8)
        (
            "deepseek-v41-flash-tp8-ep-100k-decode-dspark-kvshard.sh",
            {},
            "[8,16,24,32,48,64,96,144,192,288,384,576,768,1152,1344,1536,2048]",
        ),
        # PR #1794 dedicated C=64 DSpark offload wrapper
        (
            "deepseek-v41-flash-tp8-ep-100k-decode-c64-dspark-offload.sh",
            {},
            "[64,384,8192]",
        ),
        # Custom user override preserved across every wrapper
        (
            "deepseek-v41-flash-tp8-ep-100k-decode-dspark.sh",
            {"MAX_NUM_SEQS": "224", "COMPILE_SIZES": "[4,16,768,2048]"},
            "[4,16,768,2048]",
        ),
    ],
)
def test_compile_sizes_buckets(
    script_name: str, env_overrides: dict[str, str], expected_sizes: str
) -> None:
    result = _eval_config(script_name, env_overrides)
    assert result["COMPILE_SIZES"] == expected_sizes
    assert f'"compile_sizes":{expected_sizes}' in result["EXTRA_SERVE_ARGS"]


@pytest.mark.parametrize(
    "script_name",
    [
        "deepseek-v41-flash-tp8-ep-100k-decode.sh",
        "deepseek-v41-flash-dp2-tp4-ep-100k-decode.sh",
        "deepseek-v41-flash-tp8-ep-100k-decode-dspark.sh",
        "deepseek-v41-flash-dp2-tp4-ep-100k-decode-dspark.sh",
    ],
)
def test_max_num_batched_tokens_and_ready_wait_overrides(script_name: str) -> None:
    defaults = _eval_config(script_name, {})
    assert defaults["MAX_NUM_BATCHED_TOKENS"] == "2048"
    assert defaults["SERVER_READY_WAIT_MIN"] == "120"

    overridden = _eval_config(
        script_name,
        {"MAX_NUM_BATCHED_TOKENS": "4096", "SERVER_READY_WAIT_MIN": "90"},
    )
    assert overridden["MAX_NUM_BATCHED_TOKENS"] == "4096"
    assert overridden["SERVER_READY_WAIT_MIN"] == "90"
